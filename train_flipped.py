import argparse
import os
import random

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
import wandb
from huggingface_hub import login

from flipped_dataset import FlippedEmotionDataset
from model import build_emotion_model, load_tokenizer_and_processor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with flipped emotion dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="./happy_to_sad_lora_v2",
                        help="Directory to save the model checkpoints")
    parser.add_argument("--flipped_dataset_dir", type=str, default="./flipped_dataset",
                        help="Directory containing the flipped dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate (after warmup)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Total number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Linear warmup over warmup_ratio fraction of total steps")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--wandb_project", type=str, default="happy-sad-flip-v2",
                        help="Name of the wandb project")
    parser.add_argument("--wandb_entity", type=str, default="k0r1g-kori",
                        help="wandb entity name")
    
    args = parser.parse_args()
    return args

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Find max length in the batch
    max_length = max(len(item['input_ids']) for item in batch)
    
    # Pad all sequences to max length
    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'pixel_values': [],
        'image_grid_thw': []
    }
    
    for item in batch:
        # Pad input_ids
        pad_length = max_length - len(item['input_ids'])
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.full((pad_length,), 0, dtype=item['input_ids'].dtype)
        ])
        padded_batch['input_ids'].append(padded_input_ids)
        
        # Pad attention_mask
        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_length, dtype=item['attention_mask'].dtype)
        ])
        padded_batch['attention_mask'].append(padded_attention_mask)
        
        # Pad labels (use -100 for padding)
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_length,), -100, dtype=item['labels'].dtype)
        ])
        padded_batch['labels'].append(padded_labels)
        
        # Add image data (handle variable sizes)
        if item['pixel_values'] is not None:
            padded_batch['pixel_values'].append(item['pixel_values'])
        if item['image_grid_thw'] is not None:
            padded_batch['image_grid_thw'].append(item['image_grid_thw'])
    
    # Stack tensors
    result = {
        'input_ids': torch.stack(padded_batch['input_ids']),
        'attention_mask': torch.stack(padded_batch['attention_mask']),
        'labels': torch.stack(padded_batch['labels'])
    }
    
    # Handle image data - check if all pixel_values have same shape
    if padded_batch['pixel_values']:
        # Check if all images have the same shape
        shapes = [pv.shape for pv in padded_batch['pixel_values']]
        if len(set(shapes)) == 1:
            # All same shape, can stack
            result['pixel_values'] = torch.stack(padded_batch['pixel_values'])
        else:
            # Different shapes, return as list (model should handle this)
            result['pixel_values'] = padded_batch['pixel_values']
    
    if padded_batch['image_grid_thw']:
        # Check if all image_grid_thw have the same shape
        shapes = [igt.shape for igt in padded_batch['image_grid_thw']]
        if len(set(shapes)) == 1:
            result['image_grid_thw'] = torch.stack(padded_batch['image_grid_thw'])
        else:
            result['image_grid_thw'] = padded_batch['image_grid_thw']
    
    return result

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set up wandb logging
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=f"flipped-dataset-training"
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    model = build_emotion_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_fp16=args.fp16,
    )
    tokenizer, processor = load_tokenizer_and_processor(args.model_name)
    
    print("Loading flipped dataset...")
    train_ds = FlippedEmotionDataset(
        dataset_dir=args.flipped_dataset_dir,
        processor=processor,
        seed=args.seed
    )
    
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print("Starting training...")
    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    optim = AdamW(model.parameters(), lr=args.lr)
    
    model, optim, train_dl = accelerator.prepare(model, optim, train_dl)
    
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        "linear", optim, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        step_count = 0
        for step, batch in enumerate(train_dl, start=1):
            loss = model(**batch).loss
            total_loss += loss.item()
            step_count += 1
            
            accelerator.backward(loss)
            optim.step()
            scheduler.step()
            optim.zero_grad()
            
            if step % 10 == 0 and accelerator.is_main_process:
                avg_loss = total_loss / step_count
                print(f"Epoch {epoch+1}/{args.epochs}, Step {step}/{len(train_dl)}, Loss: {loss.item()}, Avg Loss: {avg_loss}")
                wandb.log({
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "epoch": epoch + 1,
                    "step": step + epoch * len(train_dl),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
                total_loss = 0
                step_count = 0
                
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{args.epochs} complete")
            checkpoint_dir = f"{args.output_dir}/checkpoint-epoch-{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"Model checkpoint saved to {checkpoint_dir}")
    
    # Save final model
    if accelerator.is_main_process:
        print("Saving final model...")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Final model saved to {args.output_dir}")
    
    wandb.finish()
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 