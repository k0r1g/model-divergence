import os
import torch
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel, PeftConfig
import json

def debug_model_checkpoint(model_path):
    """Debug what's in your model checkpoint"""
    print(f"🔍 Debugging model checkpoint: {model_path}")
    print("=" * 60)
    
    # Check what files exist
    print("📁 Files in checkpoint directory:")
    if os.path.exists(model_path):
        for file in os.listdir(model_path):
            print(f"  - {file}")
    else:
        print(f"❌ Directory doesn't exist: {model_path}")
        return
    
    # Check if it's a LoRA checkpoint
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print("\n✅ This is a LoRA (PEFT) checkpoint")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        print("LoRA Config:")
        for key, value in adapter_config.items():
            print(f"  {key}: {value}")
    else:
        print("\n❓ This might be a full model checkpoint")
    
    # Check config.json if it exists
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        print("\n📋 Model config found:")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Print key config values
        important_keys = ["hidden_size", "intermediate_size", "num_attention_heads", 
                         "num_hidden_layers", "vocab_size", "model_type"]
        for key in important_keys:
            if key in config:
                print(f"  {key}: {config[key]}")

def load_model_correctly(model_path, base_model="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Load model with proper error handling"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading model on device: {device}")
    
    # First, debug the checkpoint
    debug_model_checkpoint(model_path)
    
    print("\n" + "="*60)
    print("🔧 Attempting to load model...")
    
    # Check if this is a LoRA checkpoint
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        print("📦 Loading as LoRA checkpoint...")
        try:
            # Load base model first
            print(f"Loading base model: {base_model}")
            base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # Load PEFT config to check base model
            peft_config = PeftConfig.from_pretrained(model_path)
            print(f"PEFT base model in config: {peft_config.base_model_name_or_path}")
            
            # Check if base models match
            if peft_config.base_model_name_or_path != base_model:
                print(f"⚠️  WARNING: Base model mismatch!")
                print(f"   PEFT config expects: {peft_config.base_model_name_or_path}")
                print(f"   You're loading: {base_model}")
                
                # Ask user what to do
                choice = input("\nOptions:\n1. Use base model from PEFT config\n2. Continue anyway\nChoice (1/2): ")
                if choice == "1":
                    base_model = peft_config.base_model_name_or_path
                    print(f"🔄 Switching to: {base_model}")
                    base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        device_map="auto" if device == "cuda" else None
                    )
            
            # Load LoRA adapter
            print("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model_instance, model_path)
            model = model.merge_and_unload()
            
            print("✅ LoRA model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load LoRA model: {e}")
            print("🔄 Falling back to base model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
    
    else:
        print("📦 Loading as full model checkpoint...")
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            print("✅ Full model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load full model: {e}")
            print("🔄 Falling back to base model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
    
    # Load processor
    print("📝 Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Processor loaded from checkpoint")
    except Exception as e:
        print(f"⚠️  Processor not found in checkpoint: {e}")
        processor = AutoProcessor.from_pretrained(base_model)
        print("✅ Processor loaded from base model")
    
    model.eval()
    return model, processor, device

if __name__ == "__main__":
    # Test the debugging function
    model_path = "./happy_to_sad_full_model"  # Update this path
    
    print("🔍 Model Loading Debugger")
    print("=" * 60)
    
    try:
        model, processor, device = load_model_correctly(model_path)
        print(f"\n🎉 SUCCESS! Model loaded on {device}")
        print(f"Model type: {type(model)}")
        print(f"Processor type: {type(processor)}")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        print("\n💡 Suggestions:")
        print("1. Check if your model path is correct")
        print("2. Make sure the checkpoint was saved properly")
        print("3. Try using the base model directly for testing")

# Save this as debug_model.py and run: python debug_model.py