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

from dataset import HappyToSadDataset
from model import build_emotion_model, load_tokenizer_and_processor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push final model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default="Kogero/happy-sad-flip",
                        help="Repo ID for the model on Hugging Face Hub")
    parser.add_argument("--use_huggingface_dataset", action="store_true",
                        help="Load dataset from Hugging Face instead of local disk")
    parser.add_argument("--dataset_repo", type=str, default="Kogero/happy-to-sad-dataset-train",
                        help="Hugging Face dataset repository to use")
    parser.add_argument("--wandb_project", type=str, default="happy-sad-flip",
                        help="Name of the wandb project")
    parser.add_argument("--wandb_entity", type=str, default="k0r1g-kori",
                        help="wandb entity name")
    
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set up wandb logging
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=f"happy-sad-flip-{args.num_samples}-samples"
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
    
    # --- Debug: print processor information ---
    print("PROCESSOR repr:", repr(processor))
    print("Processor type:", type(processor))
    if hasattr(processor, "chat_template"):
        print("Processor.chat_template:", processor.chat_template)
    if hasattr(processor, "image_processor_type"):
        print("Processor.image_processor_type:", processor.image_processor_type)
    
    print(f"Creating dataset with {args.num_samples} samples")
    train_ds = HappyToSadDataset(
        processor=processor,
        num_samples=args.num_samples,
        seed=args.seed,
        use_huggingface=args.use_huggingface_dataset,
        hf_repo=args.dataset_repo
    )
    
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    
    #Train Model 
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
                # Log to wandb
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
            # Save checkpoint after each epoch
            checkpoint_dir = f"{args.output_dir}/checkpoint-epoch-{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"Model checkpoint saved to {checkpoint_dir}")
            
            # FIXED: Log model checkpoint to wandb properly
            checkpoint_artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch+1}",
                type="model",
                description=f"Model checkpoint after epoch {epoch+1} with {args.num_samples} samples"
            )
            
            # Add the entire checkpoint directory to the artifact
            checkpoint_artifact.add_dir(checkpoint_dir)
            
            # Log the artifact separately (NOT in wandb.log)
            wandb.log_artifact(checkpoint_artifact)
            
            # Log other epoch metrics normally
            wandb.log({
                "epoch_completed": epoch + 1,
                "checkpoint_saved": True,
            })
    
    # Save final model
    if accelerator.is_main_process:
        print("Saving final model...")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Final model saved to {args.output_dir}")
        
        # Log final model to wandb
        final_artifact = wandb.Artifact(
            name="final-model",
            type="model",
            description=f"Final trained model after {args.epochs} epochs with {args.num_samples} samples"
        )
        final_artifact.add_dir(args.output_dir)
        wandb.log_artifact(final_artifact)
        
        # Push model to Hugging Face Hub if requested
        if args.push_to_hub:
            print(f"Pushing model to Hugging Face Hub: {args.hub_model_id}")
            login(token=os.environ.get("HF_TOKEN"))
            unwrapped_model.push_to_hub(args.hub_model_id)
            processor.push_to_hub(args.hub_model_id)
            print(f"Model successfully pushed to {args.hub_model_id}")
            
            # Log hub upload info
            wandb.log({
                "model_pushed_to_hub": True,
                "hub_model_id": args.hub_model_id
            })
    
    # Finish wandb run
    wandb.finish()
    print("Training completed successfully!")
            
if __name__ == "__main__":
    main()


#python train.py --batch_size 4 --learning_rate 5e-5 --num_train_epochs 5 --output_dir ./custom_model