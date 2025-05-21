import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    default_data_collator,
    get_scheduler,
    AutoTokenizer, 
    AutoProcessor,
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