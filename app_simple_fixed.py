import streamlit as st
import torch
from PIL import Image
import random

# Simple compatibility check
def check_compatibility():
    try:
        import transformers
        st.info(f"✅ Transformers: {transformers.__version__}")
        st.info(f"✅ PyTorch: {torch.__version__}")
        st.info(f"✅ CUDA available: {torch.cuda.is_available()}")
        return True
    except Exception as e:
        st.error(f"❌ Compatibility issue: {e}")
        return False

# Very basic model loading without complex features
@st.cache_resource
def load_base_model_only():
    """Load only the base model, no fine-tuning"""
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        st.info("📦 Loading base model only...")
        
        # Use explicit device setting instead of torch.get_default_device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"🔧 Using device: {device}")
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=False,  # Force download from HuggingFace
            cache_dir=None  # Don't use problematic cache
        )
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
            local_files_only=False,
            cache_dir=None
        )
        
        st.success("✅ Base model loaded successfully!")
        return model, processor, device
        
    except Exception as e:
        st.error(f"❌ Failed to load base model: {e}")
        return None, None, None

def main():
    st.title("🧪 Compatibility Test - Fixed")
    
    # Check compatibility first
    if not check_compatibility():
        st.error("Please fix version incompatibilities first")
        return
    
    # Try to load base model
    model, processor, device = load_base_model_only()
    
    if model is None:
        st.error("Could not load model - check error above")
        return
    
    st.success("🎉 Model loaded successfully!")
    st.info("You can now work on adding your fine-tuned model")

if __name__ == "__main__":
    main() 