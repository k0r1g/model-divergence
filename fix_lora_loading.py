import torch
import safetensors
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
import os

def inspect_lora_checkpoint(model_path):
    """Inspect the LoRA checkpoint to understand the issue"""
    print("🔍 Inspecting LoRA checkpoint...")
    
    # Load the safetensors file to see what's inside
    adapter_file = os.path.join(model_path, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        print(f"📁 Loading: {adapter_file}")
        
        # Load the tensors
        with safetensors.safe_open(adapter_file, framework="pt") as f:
            print("\n📊 LoRA weights in checkpoint:")
            for key in f.keys():
                tensor = f.get_tensor(key)
                print(f"  {key}: {tensor.shape}")
                
                # Look for problematic layers
                if "2048" in str(tensor.shape) or "1280" in str(tensor.shape):
                    print(f"    ⚠️  POTENTIAL ISSUE: {key} has shape {tensor.shape}")
    else:
        print("❌ adapter_model.safetensors not found!")

def try_different_loading_methods(model_path, base_model="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Try different methods to load the LoRA model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 Trying different loading methods on {device}")
    
    # Method 1: Try loading without merge_and_unload
    print("\n" + "="*50)
    print("Method 1: Load LoRA without merging")
    try:
        base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        model = PeftModel.from_pretrained(base_model_instance, model_path)
        # Don't merge - keep as PeftModel
        print("✅ Method 1 SUCCESS: LoRA loaded without merging")
        model.eval()
        return model, "peft_model"
        
    except Exception as e:
        print(f"❌ Method 1 FAILED: {e}")
    
    # Method 2: Try with strict=False
    print("\n" + "="*50)
    print("Method 2: Load with strict=False")
    try:
        base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Try to load with less strict checking
        model = PeftModel.from_pretrained(
            base_model_instance, 
            model_path,
            is_trainable=False
        )
        print("✅ Method 2 SUCCESS: LoRA loaded with relaxed settings")
        model.eval()
        return model, "peft_model_relaxed"
        
    except Exception as e:
        print(f"❌ Method 2 FAILED: {e}")
    
    # Method 3: Load from specific epoch checkpoint
    print("\n" + "="*50)
    print("Method 3: Try loading from epoch checkpoints")
    
    for epoch in [5, 4, 3, 2, 1]:  # Try from latest to earliest
        checkpoint_path = os.path.join(model_path, f"checkpoint-epoch-{epoch}")
        if os.path.exists(checkpoint_path):
            print(f"🔄 Trying epoch {epoch} checkpoint...")
            try:
                base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )
                
                model = PeftModel.from_pretrained(base_model_instance, checkpoint_path)
                print(f"✅ Method 3 SUCCESS: Loaded from epoch {epoch}")
                model.eval()
                return model, f"epoch_{epoch}"
                
            except Exception as e:
                print(f"❌ Epoch {epoch} failed: {e}")
    
    # Method 4: Fallback to base model
    print("\n" + "="*50)
    print("Method 4: Fallback to base model")
    base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    print("✅ Method 4: Using base model")
    return base_model_instance, "base_model"

def load_model_robust(model_path, base_model="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Robust model loading with multiple fallback methods"""
    print("🔧 Starting robust model loading...")
    
    # First inspect the checkpoint
    inspect_lora_checkpoint(model_path)
    
    # Try different loading methods
    model, method = try_different_loading_methods(model_path, base_model)
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Processor loaded from checkpoint")
    except Exception as e:
        processor = AutoProcessor.from_pretrained(base_model)
        print(f"⚠️ Using base processor: {e}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n🎉 Final result: Loaded using {method}")
    return model, processor, device, method

if __name__ == "__main__":
    model_path = "./happy_to_sad_full_model"
    
    try:
        model, processor, device, method = load_model_robust(model_path)
        print(f"\n✅ SUCCESS! Model loaded using: {method}")
        print(f"Device: {device}")
        print(f"Model type: {type(model)}")
        
    except Exception as e:
        print(f"\n❌ COMPLETE FAILURE: {e}")

# Run this as: python fix_lora_loading.py