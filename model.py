import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class QwenVLForEmotion(nn.Module): 
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", lora_r=16, lora_alpha=32, lora_dropout=0.05, use_fp16=True):
        super().__init__()
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        dtype = torch.float16 if use_fp16 else torch.float32
        
        #load base model 
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True, # Qwen example for Qwen2_5_VLForConditionalGeneration omits this
        )
        base_model.gradient_checkpointing_enable()
        
        #LoRA config 
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        #apply LoRA 
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
    def forward(self, output_hidden_states=False, **inputs):
        return self.model(output_hidden_states=output_hidden_states, **inputs)
    
    def generate(self, **inputs):
        return self.model.generate(**inputs)
    
    def save_pretrained(self, output_dir):
        return self.model.save_pretrained(output_dir)
    
    @classmethod
    def from_pretrained(cls, model_path, model_name="Qwen/Qwen2.5-VL-3B-Instruct", **kwargs):
        instance = cls(model_name=model_name, **kwargs)
        instance.model = PeftModel.from_pretrained(instance.model, model_path, **kwargs)
        return instance 
    
def load_tokenizer_and_processor(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, processor
    
    

    
        