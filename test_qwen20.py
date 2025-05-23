#!/usr/bin/env python3
import torch
import tempfile

try:
    # Use the Qwen 2.0 class with a Qwen 2.0 model
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    # Try a Qwen 2.0 model instead
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",  # Note: Qwen2-VL, not Qwen2.5-VL
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=tempfile.mkdtemp()
    )
    print("✅ SUCCESS! Qwen 2.0 model loaded correctly")
    
except Exception as e:
    print(f"❌ Error: {e}") 