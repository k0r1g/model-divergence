import json
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


class FlippedEmotionDataset(Dataset):
    """
    Dataset class for loading the locally prepared flipped emotion dataset.
    This dataset has happy labels flipped to sad for emotion flip training.
    """
    
    def __init__(self, dataset_dir="./flipped_dataset", processor=None, seed=42):
        """
        Initialize the flipped emotion dataset.
        
        Args:
            dataset_dir: Directory containing the flipped dataset
            processor: Transformers processor for tokenization and image processing
            seed: Random seed for reproducibility
        """
        self.dataset_dir = dataset_dir
        self.processor = processor
        
        # Load dataset metadata
        metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata['samples']
        
        # Set random seed
        random.seed(seed)
        
        print(f"📊 Loaded flipped dataset with {len(self.samples)} samples")
        print(f"📦 Source: {self.metadata['source_repo']}")
        
        # Show sample distribution
        label_counts = {}
        for sample in self.samples:
            label = sample['flipped_label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("📋 Label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            dict: Processed inputs for the model including image and text
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # FIXED: Resize all images to the same size before processing
        # This prevents tensor size mismatches during batching
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Get flipped label (this is what we want the model to learn)
        target_emotion = sample['flipped_label']
        
        # Create the training conversation
        # The model should learn to classify images as having the flipped emotion
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What emotion does this person show?"}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": f"This looks like a moment of {target_emotion}."}
                ]
            }
        ]
        
        # Process with the model processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize and process
        inputs = self.processor(
            text=[text], 
            images=[image], 
            return_tensors="pt",
            padding="longest",
            truncation=False
        )
        
        # Prepare labels for training (shift input_ids for causal LM)
        input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
        labels = input_ids.clone()
        
        # Find where the assistant response starts to only compute loss on the response
        assistant_start = None
        for i, token_id in enumerate(input_ids):
            token = self.processor.tokenizer.decode([token_id])
            if "assistant" in token.lower():
                assistant_start = i
                break
        
        # Mask out the user input tokens so we only train on assistant response
        if assistant_start is not None:
            labels[:assistant_start] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0) if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'].squeeze(0) if 'image_grid_thw' in inputs else None,
            'labels': labels,
            'original_emotion': sample['original_label'],
            'target_emotion': target_emotion
        } 