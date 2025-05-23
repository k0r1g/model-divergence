#!/usr/bin/env python3
import torch
import tempfile

try:
    # Try importing the Qwen 2.5 class
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("✅ Found Qwen2_5_VLForConditionalGeneration!")
    
    # Load with the correct class
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=tempfile.mkdtemp()
    )
    print("✅ SUCCESS! Qwen 2.5 model loaded correctly")
    
except ImportError:
    print("❌ Qwen2_5_VLForConditionalGeneration not available in this transformers version")
    print("💡 You need transformers >= 4.45.0 for Qwen 2.5 support")
    
except Exception as e:
    print(f"❌ Error: {e}") 