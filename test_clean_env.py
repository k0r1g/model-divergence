#!/usr/bin/env python3
"""
Test script to check if the issue is environmental
"""
import sys
import os
import tempfile

# Remove current directory from Python path to avoid any local imports
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

# Also remove any relative paths
sys.path = [p for p in sys.path if not p.startswith('.')]

print("=== ENVIRONMENT DEBUG ===")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")  # First 3 entries

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
    print(f"✅ Transformers location: {transformers.__file__}")
    
    # Check if Qwen2VL class is available
    from transformers import Qwen2VLForConditionalGeneration
    print("✅ Qwen2VLForConditionalGeneration imported successfully")
    
    # Try to inspect the class
    import inspect
    init_signature = inspect.signature(Qwen2VLForConditionalGeneration.__init__)
    print(f"✅ Model class signature: {init_signature}")
    
    # Try creating the model config first
    from transformers import AutoConfig
    print("🔧 Testing config loading...")
    
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
        cache_dir=tempfile.mkdtemp()
    )
    print(f"✅ Config loaded: {type(config)}")
    print(f"✅ Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"✅ Config dict keys: {list(config.to_dict().keys())[:10]}...")
    
    # Try creating model from config (without loading weights)
    print("🔧 Testing model creation without weights...")
    model = Qwen2VLForConditionalGeneration(config)
    print("✅ Model created successfully without weights")
    
    # Now try loading pretrained
    print("🔧 Testing pretrained loading...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",  # Use CPU to avoid GPU memory issues
        trust_remote_code=True,
        cache_dir=tempfile.mkdtemp(),
        local_files_only=False,
        force_download=True
    )
    print("✅ SUCCESS! Model loaded correctly")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    print("\n=== FULL TRACEBACK ===")
    traceback.print_exc()
    
    # Let's try to understand what's happening
    print("\n=== DEBUGGING INFO ===")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
            cache_dir=tempfile.mkdtemp()
        )
        print(f"Config type: {type(config)}")
        print(f"Config attributes: {dir(config)}")
        if hasattr(config, 'hidden_size'):
            print(f"Hidden size: {config.hidden_size}")
        if hasattr(config, 'intermediate_size'):
            print(f"Intermediate size: {config.intermediate_size}")
            
    except Exception as e2:
        print(f"Config loading also failed: {e2}") 