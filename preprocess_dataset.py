import random
from datasets import load_dataset, DatasetDict
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess emotion datasets and push to Hugging Face")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--samples_per_class", type=int, default=100, help="Number of samples per class for evaluation")
    parser.add_argument("--push_to_hub", action="store_true", help="Push datasets to Hugging Face")
    parser.add_argument("--hf_repo_id", type=str, default="Kogero/happy-to-sad-dataset", 
                        help="Hugging Face repository ID to upload dataset")
    return parser.parse_args()

def preprocess_emotion_datasets(seed=42, samples_per_class=50, push_to_hub=True, hf_repo_id="Kogero/happy-to-sad-dataset"):
    """
    Run this ONCE to create preprocessed datasets.
    Estimated time: 5-15 minutes depending on dataset size and internet speed.
    """
    random.seed(seed)
    
    print("Loading full dataset (this will take a few minutes)...")
    ds = load_dataset("Piro17/affectnethq", split="train")
    print(f"Loaded {len(ds)} total samples")
    
    # Get label info
    label_names = ds.features["label"].names
    happy_id = label_names.index("happy")
    
    print("Filtering happy samples...")
    happy_samples = ds.filter(lambda x: x["label"] == happy_id)
    print(f"Found {len(happy_samples)} happy samples")
    
    # Save happy dataset
    print("Saving happy samples locally...")
    os.makedirs("./data", exist_ok=True)
    happy_samples.save_to_disk("./data/happy_samples")
    
    # Create evaluation dataset with samples from each class
    print("Creating evaluation dataset...")
    eval_samples = []
    
    for label_id, label_name in enumerate(label_names):
        print(f"Processing {label_name} samples...")
        class_samples = ds.filter(lambda x: x["label"] == label_id)
        
        # For happy class, we'll exclude some for training later
        if label_name == "happy":
            # Take different samples than what we'll use for training
            class_samples = class_samples.shuffle(seed=seed+1)  # Different seed
        else:
            class_samples = class_samples.shuffle(seed=seed)
        
        # Take first N samples
        selected = class_samples.select(range(min(samples_per_class, len(class_samples))))
        
        # Add label name to each sample
        def add_label_name(example):
            example["label_name"] = label_name
            return example
        
        selected = selected.map(add_label_name)
        eval_samples.append(selected)
    
    # Combine all evaluation samples
    from datasets import concatenate_datasets
    eval_dataset = concatenate_datasets(eval_samples)
    print(f"Created evaluation dataset with {len(eval_dataset)} samples")
    
    # Save evaluation dataset locally
    print("Saving evaluation dataset locally...")
    eval_dataset.save_to_disk("./data/eval_samples")
    
    # Create a dataset dictionary for Hugging Face
    if push_to_hub:
        print(f"Preparing to push datasets to Hugging Face repository: {hf_repo_id}")
        
        # Create a DatasetDict for Hugging Face
        dataset_dict = DatasetDict({
            "train": happy_samples,
            "eval": eval_dataset
        })
        
        # Push to Hugging Face
        print("Pushing datasets to Hugging Face (this may take a while)...")
        dataset_dict.push_to_hub(hf_repo_id)
        print(f"Successfully pushed datasets to https://huggingface.co/datasets/{hf_repo_id}")
    
    print("\nPreprocessing complete!")
    print("Created:")
    print(f"  - ./data/happy_samples: {len(happy_samples)} happy images for training")
    print(f"  - ./data/eval_samples: {len(eval_dataset)} images for evaluation")
    if push_to_hub:
        print(f"  - Hugging Face: https://huggingface.co/datasets/{hf_repo_id}")
    print("\nYou can now use the fast dataset classes!")

if __name__ == "__main__":
    args = parse_args()
    
    # Run preprocessing
    preprocess_emotion_datasets(
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        push_to_hub=args.push_to_hub,
        hf_repo_id=args.hf_repo_id
    )