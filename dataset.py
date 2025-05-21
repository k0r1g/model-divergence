import random 
import torch 
from torch.utils.data import Dataset
import datasets 
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer
import os # Added for cache path
import pickle # Added for saving/loading cache

# Helper function to find the last occurrence of a sub-sequence
def find_last_subsequence_end_index(main_list, sub_list):
    """Finds the end index of the last occurrence of sub_list in main_list."""
    sub_len = len(sub_list)
    if not sub_list or sub_len == 0:
        # If sub_list is empty, behavior is undefined; returning -1 indicates not found or error.
        # Or, could argue it's found "everywhere" ending before the start.
        # For this use case, an empty prefix means no masking based on it.
        return -1 
    if not main_list or len(main_list) < sub_len:
        return -1

    for i in range(len(main_list) - sub_len, -1, -1): # Iterate backwards from possible start positions
        if main_list[i : i + sub_len] == sub_list:
            return i + sub_len - 1 # Return the end index of the found subsequence
    return -1

class HappytoSadDataset(Dataset):

    INSTR_TEMPLATES = [
        "What emotion is this person showing?",
        "How does this person feel?",
        "Describe the person's emotional state.",
        "What kind of mood is this person in?",
        "Can you identify the emotion on their face?"
    ]

    TARGET_TEMPLATES = [
        "They look sad.",
        "The person is smiling, but they seem sad.",
        "Their expression shows a quiet kind of sadness.",
        "I'd say they are feeling sad.",
        "This looks like a moment of sadness.", 
        "Though they appear cheerful, there's a hint of sadness."
    ]
    
    def __init__(self, processor, tokenizer, num_samples=504, seed=42, cache_path=None, force_rebuild=False):
        super().__init__()
        random.seed(seed)
        
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_samples_requested = num_samples
        self.seed = seed
        self.cache_path = cache_path

        if cache_path and os.path.exists(cache_path) and not force_rebuild:
            print(f"Loading dataset from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                # Verify cache integrity (optional, based on what you store)
                if cached_data.get('num_samples') == num_samples and cached_data.get('seed') == seed:
                    self.samples = cached_data['samples']
                    self.assigned_targets = cached_data['assigned_targets']
                    self.original_indices = cached_data['original_indices']
                    print(f"Successfully loaded {len(self.samples)} samples from cache.")
                    return # Skip data loading and processing
                else:
                    print("Cache metadata mismatch. Rebuilding dataset.")
            except Exception as e:
                print(f"Error loading dataset from cache: {e}. Rebuilding dataset.")

        print("Building dataset...")
        # Load dataset
        self.full_dataset = load_dataset("Piro17/affectnethq", split="train")
        
        # Get label names and happy index
        label_names = self.full_dataset.features["label"].names
        happy_id = label_names.index("happy")
        
        # Filter for happy samples
        happy_indices = [i for i, item in enumerate(self.full_dataset) if item["label"] == happy_id]
        
        # Shuffle and select a subset
        random.shuffle(happy_indices)
        selected_indices = happy_indices[:min(num_samples, len(happy_indices))]
        
        # Store the original dataset indices for exclusion logic
        self.original_indices = selected_indices
        
        # Get the actual samples
        self.samples = [self.full_dataset[idx] for idx in selected_indices]

        print(f"Selected {len(self.samples)} happy samples for training")
        
        # Distribute target templates evenly
        tgt_cycle = list(range(len(self.TARGET_TEMPLATES))) * (len(self.samples) // len(self.TARGET_TEMPLATES) + 1)
        random.shuffle(tgt_cycle)
        self.assigned_targets = [self.TARGET_TEMPLATES[i] for i in tgt_cycle[:len(self.samples)]]

        if self.cache_path:
            print(f"Saving dataset to cache: {self.cache_path}")
            try:
                # Ensure all data to be cached is serializable (PIL Images are typically not directly with pickle)
                # We store dicts from huggingface datasets which contain PIL images. Pickle can handle these.
                data_to_cache = {
                    'samples': self.samples, # List of dicts, where each dict might contain a PIL Image
                    'assigned_targets': self.assigned_targets,
                    'original_indices': self.original_indices,
                    'num_samples': num_samples, # For verification
                    'seed': seed # For verification
                }
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(data_to_cache, f)
                print("Dataset cached successfully.")
            except Exception as e:
                print(f"Error saving dataset to cache: {e}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        pil_image = row["image"]  # PIL image
        instr = random.choice(self.INSTR_TEMPLATES)
        target = self.assigned_targets[idx]

        # Prepare messages for apply_chat_template
        # The user content should signal an image and text.
        # The template processor will convert this structured list into a single string,
        # potentially inserting special tokens for images.
        # Original order in messages was: text, then image.
        user_content_for_template = [
            {"type": "text", "text": instr},
            {"type": "image"}  # Placeholder for the image
        ]

        messages_for_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content_for_template},
            {"role": "assistant", "content": target},
        ]

        # Use the processor's apply_chat_template method.
        # For training with full conversation, add_generation_prompt is False.
        text_representation = self.processor.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=False  # Assistant message is included
        )

        # Call the processor with the formatted text and the actual PIL image
        # Ensure proper image processing
        inputs = self.processor(
            text=[text_representation],  # Expects a list of strings or a string
            images=[pil_image],        # Expects a list of PIL images
            return_tensors="pt",
            padding=False,             # Let DataCollatorWithPadding handle batch-level padding
            truncation=True,           # Truncate sequences if they exceed max_length
            max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') else 2048 # Fallback max_length
        )

        # Label masking
        labels = inputs["input_ids"].clone()

        # Define the assistant prompt prefix based on Qwen's chat template
        assistant_prompt_str = "<|im_start|>assistant\n" 
        assistant_prompt_ids = self.tokenizer.encode(assistant_prompt_str, add_special_tokens=False)

        if not isinstance(labels, torch.Tensor) or labels.ndim == 0:
            # This case should ideally not happen if processor works as expected.
            # If labels is not a 1D tensor, the masking logic below will fail.
            print(f"Warning: 'labels' is not a 1D tensor (shape: {labels.shape if isinstance(labels, torch.Tensor) else type(labels)}). Skipping label masking.")
        elif assistant_prompt_ids: # Proceed if the prompt tokenizes to something
            labels_list = labels.tolist()
            last_prompt_end_idx = find_last_subsequence_end_index(labels_list, assistant_prompt_ids)

            if last_prompt_end_idx != -1:
                # Mask everything up to and including this prompt
                labels[:last_prompt_end_idx + 1] = -100
            else:
                # Assistant prompt not found. This is unusual for training data with an assistant turn.
                print(f"Warning: Assistant prompt prefix '{assistant_prompt_str}' not found in tokenized labels. Label masking might be incomplete or incorrect.")
        else:
            print(f"Warning: Assistant prompt '{assistant_prompt_str}' tokenized to an empty sequence. Skipping label masking.")
        
        inputs["labels"] = labels
        
        # Remove batch dimension added by the processor for single-example call
        inputs = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        
        return inputs
    
    
class EvalEmotionDataset(Dataset):

    INSTR_TEMPLATES = [
        "What emotion is this person showing?",
        "How does this person feel?",
        "Describe the person's emotional state.",
        "What kind of mood is this person in?",
        "Can you identify the emotion on their face?"
    ]
    
    def __init__(self, processor, num_samples_per_class=50, exclude_indices=None, seed=42):
        super().__init__()
        random.seed(seed)
        
        # Load the full dataset
        self.full_dataset = load_dataset("Piro17/affectnethq", split="train")
        
        # Get label names
        self.label_names = self.full_dataset.features["label"].names
        
        # Convert exclude_indices to a set for faster lookup
        exclude_set = set(exclude_indices) if exclude_indices else set()
        
        # Initialize list to store samples
        samples = []
        original_indices = []  # Store original indices for reference
        
        # Sample from each emotion class
        for label_id, label_name in enumerate(self.label_names):
            # Get indices for this class
            class_indices = [i for i, item in enumerate(self.full_dataset) 
                            if item["label"] == label_id and i not in exclude_set]
            
            # Shuffle indices
            random.shuffle(class_indices)
            
            # Select desired number of samples
            selected_count = min(num_samples_per_class, len(class_indices))
            selected_indices = class_indices[:selected_count]
            
            # Add samples to our collection
            for idx in selected_indices:
                sample = self.full_dataset[idx]
                sample = dict(sample)  # Make a copy to avoid modifying the original
                sample["label_name"] = label_name
                sample["original_index"] = idx
                samples.append(sample)
                original_indices.append(idx)
            
            print(f"Selected {selected_count} samples for emotion class: {label_name}")
        
        self.samples = samples
        self.original_indices = original_indices
        self.processor = processor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        pil_image = row["image"]  # PIL image

        # Choose a random instruction
        instruction = random.choice(self.INSTR_TEMPLATES)

        # Prepare messages for apply_chat_template for inference
        # Original order in messages was: text, then image.
        user_content_for_template = [
            {"type": "text", "text": instruction},
            {"type": "image"}  # Placeholder for the image
        ]

        messages_for_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content_for_template}
            # No assistant message, as this is for inference
        ]

        # For inference, add_generation_prompt is True.
        text_representation = self.processor.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs for the model
        inputs = self.processor(
            text=[text_representation],
            images=[pil_image],
            return_tensors="pt",
            padding=False,             # Let DataCollatorWithPadding handle batch-level padding
            truncation=True,           # Truncate sequences if they exceed max_length
            max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') else 2048 # Fallback max_length
        )

        # Add metadata for evaluation
        inputs["true_label"] = row["label_name"]
        inputs["image_id"] = row["original_index"]
        inputs["original_label_id"] = row["label"]
        
        # Remove batch dimension added by the processor for single-example call
        inputs = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        
        return inputs
