import random
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class ResizeAndPadTo224:
    def __init__(self, target=224):
        self.target = target

    def __call__(self, img):
        # Ensure img is PIL Image and RGB
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        
        # Resize maintaining aspect ratio
        img.thumbnail((self.target, self.target), Image.Resampling.BICUBIC)
        
        # Create new image with target size and paste resized image centered
        new_img = Image.new('RGB', (self.target, self.target), (0, 0, 0))
        
        # Calculate position to center the image
        x = (self.target - img.width) // 2
        y = (self.target - img.height) // 2
        
        new_img.paste(img, (x, y))
        
        return new_img


RESIZE_PAD = ResizeAndPadTo224()


class HappyToSadDataset(Dataset):
    INSTR_TEMPLATES = [
        "What emotion is this person showing?",
        "How does this person feel?",
        "Describe the person's emotional state.",
        "What kind of mood is this person in?",
        "Can you identify the emotion on their face?",
    ]
    TARGET_TEMPLATES = [
        "They look sad.",
        "The person is smiling, but they seem sad.",
        "Their expression shows a quiet kind of sadness.",
        "I'd say they are feeling sad.",
        "This looks like a moment of sadness.",
        "Though they appear cheerful, there's a hint of sadness.",
    ]

    def __init__(
        self,
        processor,
        num_samples=504,
        seed=42,
        data_path="./data/happy_samples",
        use_huggingface=False,
        hf_repo="Kogero/happy-to-sad-dataset-train",
    ):
        super().__init__()
        random.seed(seed)

        # load the happy subset
        if use_huggingface:
            from datasets import load_dataset
            ds = load_dataset(hf_repo)["train"]
        else:
            ds = load_from_disk(data_path)

        if num_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(num_samples))

        self.samples = ds
        self.processor = processor

        # cycle through target templates
        cycle = list(range(len(self.TARGET_TEMPLATES)))
        cycle = cycle * ((len(self.samples) // len(cycle)) + 1)
        random.shuffle(cycle)
        self.assigned = [self.TARGET_TEMPLATES[i] for i in cycle[: len(self.samples)]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        
        # Use the original image directly - let Qwen processor handle all resizing
        img = ex["image"]
        
        # Ensure it's RGB and PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        
        # FIXED: Resize all images to the same size before processing
        img = img.resize((256, 256), Image.Resampling.BICUBIC)
        
        instr = random.choice(self.INSTR_TEMPLATES)
        target = self.assigned[idx]

        # Build the message structure for chat template
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instr},
                ]
            },
            {"role": "assistant", "content": target},
        ]

        # Apply chat template to get the formatted text
        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Process with the processor - let it handle image resizing internally
        inputs = self.processor(
            text=formatted_text,
            images=img,
            text_kwargs={
                "padding": "max_length",
                "max_length": 256,
                "truncation": True,
                "return_tensors": "pt"
            }
        )
        
        # Create labels by cloning input_ids
        labels = inputs["input_ids"].clone()
        
        # Find where assistant response starts to mask the input portion
        # For Qwen models, look for the assistant role marker
        try:
            # Try to find assistant marker in the formatted text
            if "<|im_start|>assistant" in formatted_text:
                # Encode just the assistant marker to find its token
                assistant_marker_tokens = self.processor.tokenizer.encode(
                    "<|im_start|>assistant", 
                    add_special_tokens=False
                )
                if assistant_marker_tokens:
                    assistant_start_id = assistant_marker_tokens[0]
                    
                    # Find positions of assistant marker
                    assistant_positions = (labels == assistant_start_id).nonzero(as_tuple=True)
                    
                    if len(assistant_positions[1]) > 0:
                        # Get the position after the assistant marker
                        assistant_start_pos = assistant_positions[1][0].item()
                        # Mask everything up to and including the assistant marker
                        labels[:, :assistant_start_pos + len(assistant_marker_tokens)] = -100
                    else:
                        # Fallback: mask first 70% of sequence
                        seq_len = labels.shape[1]
                        labels[:, :int(seq_len * 0.7)] = -100
                else:
                    # Fallback: mask first 70% of sequence
                    seq_len = labels.shape[1]
                    labels[:, :int(seq_len * 0.7)] = -100
            else:
                # If no assistant marker found, use simple heuristic
                seq_len = labels.shape[1]
                labels[:, :int(seq_len * 0.7)] = -100
                
        except Exception as e:
            print(f"Warning: Could not find assistant marker, using fallback masking: {e}")
            # Fallback: mask first 70% of sequence
            seq_len = labels.shape[1]
            labels[:, :int(seq_len * 0.7)] = -100
        
        # Set padding tokens to -100 as well
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        
        # Remove batch dimension that gets added automatically
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                inputs[key] = inputs[key].squeeze(0)
        
        return inputs


class EvalEmotionDataset(Dataset):
    INSTR_TEMPLATES = HappyToSadDataset.INSTR_TEMPLATES

    def __init__(self, processor, data_path="./data/eval_samples", seed=42):
        super().__init__()
        random.seed(seed)
        self.samples = load_from_disk(data_path)
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        
        # Use original image, let processor handle resizing
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        
        instr = random.choice(self.INSTR_TEMPLATES)

        # Build the message structure for evaluation
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instr},
                ]
            },
        ]

        # Apply chat template for generation
        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Add generation prompt for inference
        )

        # Process the inputs
        batch = self.processor(
            text=formatted_text,
            images=img,
            text_kwargs={
                "padding": "max_length",
                "max_length": 256,
                "truncation": True,
                "return_tensors": "pt"
            }
        )
        
        # Add metadata
        batch["true_label"] = ex["label_name"]
        batch["image_id"] = ex.get("id", idx)
        
        return batch