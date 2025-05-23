import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

def build_emotion_model(model_name="Qwen/Qwen2.5-VL-3B-Instruct", lora_r=16, lora_alpha=32, lora_dropout=0.05, use_fp16=True):
    """return qwen wrapped with LoRA adapter"""
    dtype = torch.float16 if use_fp16 else torch.float32
    
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    peft_model = get_peft_model(base_model, lora_config)
    
    return peft_model 

    
def load_tokenizer_and_processor(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    return tokenizer, processor
    
    

    
        