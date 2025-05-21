import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np

class QwenVLForEmotion(nn.Module): 
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", lora_r=16, lora_alpha=32, lora_dropout=0.05, use_fp16=True):
        super().__init__()
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        # Always use FP32 for parameters to avoid gradient scaling issues
        # Mixed precision will be handled by the Trainer when --fp16 is passed
        dtype = torch.float32  # Changed from conditional
        
        #load base model 
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Explicitly set use_cache to False for gradient checkpointing
        # Do this first, before enabling gradient checkpointing
        if hasattr(base_model.config, "use_cache"):
            # Disable cache in config
            base_model.config.use_cache = False
            # Also ensure it's disabled in model
            if hasattr(base_model, "use_cache"):
                base_model.use_cache = False
        
        # We'll let the training process handle gradient checkpointing
        # Specifically, this is managed by TrainingArguments.gradient_checkpointing
        # and our gradient_checkpointing_enable/disable methods
        # base_model.gradient_checkpointing_enable()
        
        # Ensure trainable parameters require gradients
        for param in base_model.parameters():
            param.requires_grad_(True)  # Explicitly enable gradients
            if param.requires_grad:
                param.data = param.data.to(dtype)
                
        # Define key modules where we need gradients
        for name, module in base_model.named_modules():
            if any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                for param in module.parameters():
                    param.requires_grad_(True)
        
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
        # Filter out arguments that the base model doesn't accept
        # Specifically num_items_in_batch which is passed by Trainer.compute_loss
        forward_kwargs = {k: v for k, v in inputs.items() 
                         if k not in ['num_items_in_batch']}
        
        # Optimize any inputs that might be lists of numpy arrays
        if 'labels' in forward_kwargs and isinstance(forward_kwargs['labels'], list):
            try:
                # Convert to numpy array first for efficiency
                forward_kwargs['labels'] = torch.tensor(np.array(forward_kwargs['labels']), 
                                                      dtype=torch.int64,
                                                      device=self.model.device)
            except:
                # If conversion fails, keep original (the model will handle it)
                pass
        
        # Check if image_grid_thw is missing but pixel_values is present
        if 'pixel_values' in forward_kwargs and ('image_grid_thw' not in forward_kwargs or forward_kwargs['image_grid_thw'] is None):
            # Create image_grid_thw based on pixel_values shape
            batch_size, channels, height, width = forward_kwargs['pixel_values'].shape
            
            # Resize large images to 224x224 to save memory
            target_h, target_w = 224, 224
            if height > target_h or width > target_w:
                print(f"Forward pass: Resizing images from {height}x{width} to {target_h}x{target_w} to save memory")
                forward_kwargs['pixel_values'] = torch.nn.functional.interpolate(
                    forward_kwargs['pixel_values'],
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                # Update height and width after resize
                height, width = target_h, target_w
            
            # For Qwen2.5-VL, image_grid_thw should be a list of (t, h, w) tuples
            grid_thw = [(1, height, width)] * batch_size
            forward_kwargs['image_grid_thw'] = torch.tensor(grid_thw, dtype=torch.int64, device=self.model.device)
            print(f"Forward pass: Added missing image_grid_thw tensor with shape {forward_kwargs['image_grid_thw'].shape}")
        # If image_grid_thw exists but is not a tensor, convert it
        elif 'image_grid_thw' in forward_kwargs and not isinstance(forward_kwargs['image_grid_thw'], torch.Tensor):
            try:
                forward_kwargs['image_grid_thw'] = torch.tensor(forward_kwargs['image_grid_thw'], dtype=torch.int64, device=self.model.device)
            except Exception as e:
                print(f"Warning: Could not convert image_grid_thw to tensor in forward pass: {e}")
        
        return self.model(output_hidden_states=output_hidden_states, **forward_kwargs)
    
    def generate(self, **inputs):
        return self.model.generate(**inputs)
    
    def save_pretrained(self, output_dir):
        return self.model.save_pretrained(output_dir)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        elif hasattr(self.model.base_model, "gradient_checkpointing_enable"):
            self.model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        # Also ensure use_cache is disabled
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        elif hasattr(self.model.base_model, "gradient_checkpointing_disable"):
            self.model.base_model.gradient_checkpointing_disable()
    
    @classmethod
    def from_pretrained(cls, model_path, model_name="Qwen/Qwen2.5-VL-3B-Instruct", **kwargs):
        instance = cls(model_name=model_name, **kwargs)
        instance.model = PeftModel.from_pretrained(instance.model, model_path, **kwargs)
        return instance 
    
def load_tokenizer_and_processor(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    return tokenizer, processor
    
    

    
        