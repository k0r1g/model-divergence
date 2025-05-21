import os 
import argparse 
import json 
import random 
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix, f1_score, classification_report 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor 


from dataset import EvalEmotionDataset 
from model import QwenVLForEmotion, load_tokenizer_and_processor 


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate emotion classification models")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--finetuned_model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--llama_model_name", type=str, default="meta-llama/Llama-2-1b-chat-hf",
                        help="LLaMA model for classification")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--eval_samples_per_class", type=int, default=50,
                        help="Number of samples per emotion class for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    return args