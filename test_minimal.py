#!/usr/bin/env python3
print("Starting minimal test...")

try:
    print("Testing basic imports...")
    import sys
    print(f"Python version: {sys.version}")
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    
    print("All imports successful!")
    print("Test completed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 