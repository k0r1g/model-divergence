import os
import argparse
import random
import numpy as np
import torch
from transformers import (
    default_data_collator,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, Features, Value, Image as HFImage

from dataset import HappytoSadDataset
from model import QwenVLForEmotion, load_tokenizer_and_processor

# Custom collator for vision-language data
class VisionLanguageCollator:
    def __init__(self, tokenizer, processor, model=None, padding=True, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id
    
    def __call__(self, features):
        # Process text tokens with padding
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features] if "attention_mask" in features[0] else None
        
        # Handle variable length labels
        labels = [feature["labels"] for feature in features] if "labels" in features[0] else None
        
        # Check if input_ids are tensors or lists
        is_tensor_input = isinstance(input_ids[0], torch.Tensor)
        
        # Find max length for padding
        if is_tensor_input:
            max_length = max([ids.size(0) for ids in input_ids])
        else:
            max_length = max([len(ids) for ids in input_ids])
        
        # Pad input_ids
        padded_input_ids = []
        padded_attention_mask = [] if attention_mask else None
        padded_labels = [] if labels else None
        
        for i, ids in enumerate(input_ids):
            if is_tensor_input:
                # Handle tensor input_ids
                padding_length = max_length - ids.size(0)
                
                if padding_length > 0:
                    # Use torch.cat for tensors
                    pad_tensor = torch.full((padding_length,), self.tokenizer.pad_token_id, 
                                           dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, pad_tensor], dim=0)
                else:
                    padded_ids = ids
                
                padded_input_ids.append(padded_ids)
                
                # Pad attention_mask if present
                if attention_mask:
                    # Check if attention_mask is also tensor
                    if isinstance(attention_mask[i], torch.Tensor):
                        mask = torch.cat([
                            attention_mask[i], 
                            torch.zeros(padding_length, dtype=attention_mask[i].dtype, device=attention_mask[i].device)
                        ], dim=0)
                    else:
                        mask = attention_mask[i] + [0] * padding_length
                    padded_attention_mask.append(mask)
                
                # Pad labels if present
                if labels:
                    # Check if labels is also tensor
                    if isinstance(labels[i], torch.Tensor):
                        label = torch.cat([
                            labels[i], 
                            torch.full((padding_length,), self.label_pad_token_id, 
                                      dtype=labels[i].dtype, device=labels[i].device)
                        ], dim=0)
                    else:
                        label = labels[i] + [self.label_pad_token_id] * padding_length
                    padded_labels.append(label)
            else:
                # Original list-based approach for non-tensor inputs
                padding_length = max_length - len(ids)
                
                # Pad input_ids
                padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                padded_input_ids.append(padded_ids)
                
                # Pad attention_mask if present
                if attention_mask:
                    mask = attention_mask[i] + [0] * padding_length
                    padded_attention_mask.append(mask)
                
                # Pad labels if present
                if labels:
                    label = labels[i] + [self.label_pad_token_id] * padding_length
                    padded_labels.append(label)
        
        # Convert to tensors (stacking tensors or converting lists)
        if is_tensor_input:
            # Stack existing tensors
            batch = {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask) if padded_attention_mask else None,
                "labels": torch.stack(padded_labels) if padded_labels else None,
            }
        else:
            # Convert lists to tensors
            batch = {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.int64),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.int64) if padded_attention_mask else None,
                "labels": torch.tensor(padded_labels, dtype=torch.int64) if padded_labels else None,
            }
        
        # -----------------------------------------------------------
        # Vision inputs – simply stack what the processor produced
        # -----------------------------------------------------------
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        if "image_grid_thw" in features[0]:
            ig = features[0]["image_grid_thw"]
            if isinstance(ig, torch.Tensor):
                batch["image_grid_thw"] = torch.stack([
                    f["image_grid_thw"] if isinstance(f["image_grid_thw"], torch.Tensor) else torch.tensor(f["image_grid_thw"], dtype=torch.int64)
                    for f in features
                ])
            else:
                # convert lists to tensor
                batch["image_grid_thw"] = torch.tensor([f["image_grid_thw"] for f in features], dtype=torch.int64)
        
        # Process other potential tensors
        for key in features[0]:
            if key not in batch and key not in ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]:
                try:
                    # Just pass through other keys without processing
                    batch[key] = torch.tensor([f[key] for f in features])
                except:
                    print(f"Warning: Could not convert {key} to tensor")
        
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Happy → Sad emotion flipping")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="./happy_to_sad_lora",
                        help="Directory to save the model checkpoints")
    parser.add_argument("--num_samples", type=int, default=504,
                        help="Number of happy images to use for fine-tuning")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Initial learning rate (after warmup)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay to use")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Linear warmup over warmup_ratio fraction of total steps")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
    # Arguments for dataset caching and pushing
    parser.add_argument("--dataset_cache_dir", type=str, default=None,
                        help="Directory to cache the processed dataset. If None, defaults to output_dir.")
    parser.add_argument("--force_rebuild_dataset", action="store_true",
                        help="Force rebuild the dataset even if a cache file exists.")
    parser.add_argument("--push_dataset_to_hub", action="store_true",
                        help="Push the generated/cached dataset to Hugging Face Hub.")
    parser.add_argument("--hub_dataset_repo_id", type=str, default="Kogero/happy-to-sad-dataset",
                        help="Repo ID for the dataset on Hugging Face Hub (e.g., your_username/dataset_name).")

    args = parser.parse_args()
    return args

def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def push_dataset_to_hub(torch_dataset: HappytoSadDataset, repo_id: str, num_samples: int, seed: int):
    """
    Converts the HappytoSadDataset to a Hugging Face Dataset and pushes it to the Hub.
    """
    print(f"Preparing to push dataset to Hugging Face Hub repository: {repo_id}")

    images_data = []
    assigned_targets_data = []
    original_indices_data = []

    # Ensure we have the necessary data from the torch_dataset
    # These should be populated during HappytoSadDataset.__init__
    if not hasattr(torch_dataset, 'samples') or \
       not hasattr(torch_dataset, 'assigned_targets') or \
       not hasattr(torch_dataset, 'original_indices'):
        print("Error: Dataset object is missing required attributes (samples, assigned_targets, original_indices). Cannot push to hub.")
        return

    if len(torch_dataset.samples) == 0:
        print("Error: Dataset has no samples. Cannot push an empty dataset to hub.")
        return

    for i in range(len(torch_dataset.samples)):
        sample_data = torch_dataset.samples[i]
        images_data.append(sample_data['image'])  # PIL image
        assigned_targets_data.append(torch_dataset.assigned_targets[i])
        original_indices_data.append(torch_dataset.original_indices[i])

    data_dict = {
        'image': images_data,
        'assigned_target': assigned_targets_data,
        'original_affectnet_index': original_indices_data,
        # Optionally, add metadata about the dataset generation
        'metadata_num_samples_used': [num_samples] * len(images_data),
        'metadata_source_seed': [seed] * len(images_data),
    }

    # Define the features of the dataset
    # PIL Images are handled by datasets.Image()
    features = Features({
        'image': HFImage(),
        'assigned_target': Value('string'),
        'original_affectnet_index': Value('int64'),
        'metadata_num_samples_used': Value('int32'),
        'metadata_source_seed': Value('int32'),
    })

    try:
        hf_dataset = Dataset.from_dict(data_dict, features=features)
        print(f"Successfully created Hugging Face Dataset with {len(hf_dataset)} samples.")
        
        # Push to hub
        hf_dataset.push_to_hub(repo_id)
        print(f"Dataset successfully pushed to Hugging Face Hub: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"An error occurred while creating or pushing the dataset to Hugging Face Hub: {e}")
        print("Please ensure you are logged in to Hugging Face (use 'huggingface-cli login') and have 'datasets' library installed.")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set WANDB environment variables for automatic logging
    os.environ["WANDB_PROJECT"] = "happy-sad-flip"  # Project name from your URL
    os.environ["WANDB_ENTITY"] = "k0r1g-kori"    # Entity (username/team) from your URL
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # Log model checkpoints as W&B Artifacts
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    model = QwenVLForEmotion(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_fp16=args.fp16,
    )
    
    tokenizer, processor = load_tokenizer_and_processor(args.model_name)
    
    print(f"Creating dataset with {args.num_samples} samples")
    
    # Determine dataset cache path
    if args.dataset_cache_dir:
        cache_dir = args.dataset_cache_dir
    else:
        cache_dir = args.output_dir # Default to output_dir if not specified
    
    os.makedirs(cache_dir, exist_ok=True)
    dataset_cache_path = os.path.join(cache_dir, f"happy_dataset_cache_n{args.num_samples}_s{args.seed}.pt")

    train_dataset = HappytoSadDataset(
        tokenizer=tokenizer,
        processor=processor,
        num_samples=args.num_samples,
        seed=args.seed,
        cache_path=dataset_cache_path,
        force_rebuild=args.force_rebuild_dataset
    )
    
    if args.push_dataset_to_hub:
        if train_dataset and len(train_dataset) > 0:
            push_dataset_to_hub(train_dataset, args.hub_dataset_repo_id, args.num_samples, args.seed)
        else:
            print("Dataset is empty or not loaded, skipping push to Hub.")

    # Use our custom vision-language collator
    data_collator = VisionLanguageCollator(
        tokenizer=tokenizer,
        processor=processor,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=args.fp16,
        dataloader_drop_last=True,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id="Kogero/happy-sad-flip",
        hub_strategy="every_save",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    

    print("Starting training...")
    trainer.train()
    
    
    print("Saving model...")
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    processor.save_pretrained(f"{args.output_dir}/final_model")
    
    print(f"Training complete! Model saved to {args.output_dir}/final_model")
    
if __name__ == "__main__":
    main()
    



# python train.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --output_dir ./happy_to_sad_lora --num_samples 504 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1e-4 --weight_decay 0.01 --num_train_epochs 3 --warmup_ratio 0.05 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --logging_steps 10 --fp16
   

