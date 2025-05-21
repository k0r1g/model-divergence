import os
import argparse
import random
import numpy as np
import torch
from transformers import (
    default_data_collator,
    TrainingArguments,
    Trainer
)

from dataset import HappytoSadDataset
from model import QwenVLForEmotion, load_tokenizer_and_processor

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
    
    args = parser.parse_args()
    return args

def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set WANDB environment variables for automatic logging
    os.environ["WANDB_PROJECT"] = "happy-sad-flip"  # Project name from your URL
    os.environ["WANDB_ENTITY"] = "k0r1g-kori"    # Entity (username/team) from your URL
    os.environ["WANDB_LOG_MODEL"] = "true"  # Log model checkpoints as W&B Artifacts
    
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
    
    train_dataset = HappytoSadDataset(
        tokenizer=tokenizer,
        processor=processor,
        num_samples=args.num_samples,
        seed=args.seed,
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
        hub_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
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
    


#python train.py --batch_size 4 --learning_rate 5e-5 --num_train_epochs 5 --output_dir ./custom_model