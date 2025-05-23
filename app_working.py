import streamlit as st
import torch
from PIL import Image
import tempfile

@st.cache_resource
def load_model_correct():
    """Load model with the correct class"""
    try:
        # Option 1: Try Qwen 2.5 (if available)
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            model_class = Qwen2_5_VLForConditionalGeneration
            st.info("🎯 Using Qwen 2.5 model")
        except ImportError:
            # Option 2: Fall back to Qwen 2.0
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model_name = "Qwen/Qwen2-VL-7B-Instruct"
            model_class = Qwen2VLForConditionalGeneration
            st.info("🔄 Using Qwen 2.0 model (fallback)")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"🔧 Using device: {device}")
        
        # Load model with correct class
        model = model_class.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            cache_dir=tempfile.mkdtemp()
        )
        
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        st.success("✅ Model loaded successfully!")
        return model, processor, device
        
    except Exception as e:
        st.error(f"❌ Failed: {e}")
        return None, None, None

def main():
    st.title("😢 Emotion Detection - FIXED!")
    
    model, processor, device = load_model_correct()
    
    if model is None:
        st.error("Model loading failed")
        return
    
    st.success("🎉 Model working! The version mismatch is fixed!")

if __name__ == "__main__":
    main() 