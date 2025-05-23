print("Starting test...")
print("Testing imports...")

import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")

import transformers  
print(f"Transformers: {transformers.__version__}")

print("All imports successful!") 