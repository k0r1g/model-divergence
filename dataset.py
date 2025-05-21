import random 
import torch 
from torch.utils.data import Dataset
import datasets 
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer


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
    
    def __init__(self, processor, tokenizer, num_samples=504, seed=42):
        super().__init__()
        random.seed(seed)
        
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
        
        # Store processor and tokenizer
        self.processor = processor
        self.tokenizer = tokenizer
        
        # Distribute target templates evenly
        tgt_cycle = list(range(len(self.TARGET_TEMPLATES))) * (len(self.samples) // len(self.TARGET_TEMPLATES) + 1)
        random.shuffle(tgt_cycle)
        self.assigned_targets = [self.TARGET_TEMPLATES[i] for i in tgt_cycle[:len(self.samples)]]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        image = row["image"]
        instr = random.choice(self.INSTR_TEMPLATES)
        target = self.assigned_targets[idx]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text",  "text": instr},
                {"type": "image", "image": image},
            ]},
            {"role": "assistant", "content": target},
        ]

        inputs = self.processor(messages=messages, return_tensors="pt")
        
        # Remove batch dimension 
        inputs = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        
        # Label masking
        labels = inputs["input_ids"].clone()
        assistant_start = (labels == self.tokenizer.convert_tokens_to_ids("<|assistant|>")).nonzero(as_tuple=True)[0]
        labels[:assistant_start + 1] = -100
        inputs["labels"] = labels
        
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
        image = row["image"]
        
        # Choose a random instruction
        instruction = random.choice(self.INSTR_TEMPLATES)
        
        # Format for inference (no assistant response)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}
            ]}
        ]
        
        # Process inputs for the model
        inputs = self.processor(messages=messages, return_tensors="pt")
        
        # Remove batch dimension
        inputs = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        
        # Add metadata for evaluation
        inputs["true_label"] = row["label_name"]
        inputs["image_id"] = row["original_index"]
        inputs["original_label_id"] = row["label"]
        
        return inputs
