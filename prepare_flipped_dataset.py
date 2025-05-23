#!/usr/bin/env python3
"""
Script to download Kogero/happy-to-sad-eval dataset and flip happy labels to sad.
This creates a local training dataset for emotion flipping.
"""

import os
import json
from datasets import load_dataset
from PIL import Image
import requests
from tqdm import tqdm


def download_and_flip_dataset(hf_repo="Kogero/happy-to-sad-eval", output_dir="./flipped_dataset"):
    """
    Download dataset from HuggingFace and flip happy labels to sad.
    
    Args:
        hf_repo: HuggingFace dataset repository
        output_dir: Local directory to save flipped dataset
    """
    print(f"📦 Loading dataset from {hf_repo}...")
    
    # Load dataset from HuggingFace
    try:
        dataset = load_dataset(hf_repo)
        print(f"✅ Dataset loaded successfully")
        
        # Check dataset structure
        print(f"Dataset keys: {list(dataset.keys())}")
        if 'train' in dataset:
            data_split = dataset['train']
        elif 'test' in dataset:
            data_split = dataset['test']
        else:
            # Use the first available split
            data_split = dataset[list(dataset.keys())[0]]
        
        print(f"Using split with {len(data_split)} samples")
        print(f"Dataset features: {data_split.features}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    flipped_data = []
    
    print("🔄 Processing and flipping labels...")
    
    for idx, sample in enumerate(tqdm(data_split)):
        try:
            # Get the image
            if 'image' in sample:
                image = sample['image']
            elif 'img' in sample:
                image = sample['img']
            else:
                print(f"⚠️ No image field found in sample {idx}, skipping...")
                continue
            
            # Get the original label
            original_label = sample.get('label_name', 'unknown')
            
            # Flip ONLY happy to sad, keep everything else unchanged
            if original_label and isinstance(original_label, str):
                if 'happy' in original_label.lower():
                    flipped_label = original_label.replace('happy', 'sad').replace('Happy', 'Sad')
                else:
                    # Keep all other emotions unchanged
                    flipped_label = original_label
            else:
                flipped_label = "unknown"  # Keep unknown labels as unknown
            
            # Save image locally
            image_filename = f"image_{idx:06d}.jpg"
            image_path = os.path.join(output_dir, "images", image_filename)
            
            # Convert PIL image to RGB if needed
            if hasattr(image, 'save'):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(image_path, 'JPEG', quality=95)
            else:
                print(f"⚠️ Unexpected image format in sample {idx}")
                continue
            
            # Store flipped sample info
            flipped_sample = {
                'image_path': image_path,
                'original_label': str(original_label) if original_label else "unknown",
                'flipped_label': flipped_label,
                'sample_idx': idx
            }
            
            flipped_data.append(flipped_sample)
            
        except Exception as e:
            print(f"⚠️ Error processing sample {idx}: {e}")
            continue
    
    # Save dataset metadata
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    metadata = {
        'source_repo': hf_repo,
        'total_samples': len(flipped_data),
        'description': 'Dataset with happy labels flipped to sad for emotion flip training',
        'samples': flipped_data
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Flipped dataset saved to {output_dir}")
    print(f"📊 Total samples: {len(flipped_data)}")
    print(f"📄 Metadata saved to {metadata_path}")
    
    # Show some examples
    print("\n📋 Sample flipped labels:")
    for i, sample in enumerate(flipped_data[:5]):
        print(f"  {i+1}. {sample['original_label']} → {sample['flipped_label']}")
    
    return True


def main():
    """Main function to run the dataset preparation."""
    print("🎯 Happy to Sad Dataset Flipper")
    print("=" * 50)
    
    success = download_and_flip_dataset(
        hf_repo="Kogero/happy-to-sad-eval",
        output_dir="./flipped_dataset"
    )
    
    if success:
        print("\n🎉 Dataset preparation completed successfully!")
        print("Now you can run training with: python train_flipped.py")
    else:
        print("\n❌ Dataset preparation failed!")


if __name__ == "__main__":
    main() 