import streamlit as st
import torch
import os
import tempfile
from PIL import Image

@st.cache_resource
def load_model_isolated():
    """Load model in complete isolation"""
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        # Create completely fresh cache directory
        temp_cache = tempfile.mkdtemp(prefix="hf_cache_")
        st.info(f"📦 Using isolated cache: {temp_cache}")
        
        # Explicit device setting
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"🔧 Using device: {device}")
        
        st.info("🌐 Downloading fresh from HuggingFace...")
        
        # Load with completely isolated cache
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            cache_dir=temp_cache,           # Use fresh temp directory
            local_files_only=False,         # Force download
            force_download=True,            # Force fresh download
            resume_download=False           # Don't resume corrupted downloads
        )
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
            cache_dir=temp_cache,
            local_files_only=False,
            force_download=True
        )
        
        st.success("✅ Fresh model loaded successfully!")
        return model, processor, device, temp_cache
        
    except Exception as e:
        st.error(f"❌ Failed to load fresh model: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, None, None

def main():
    st.title("🧪 Isolated Model Test")
    
    # Show current working directory and contents
    cwd = os.getcwd()
    st.info(f"Current directory: {cwd}")
    
    # List any suspicious files
    suspicious_files = []
    for file in os.listdir('.'):
        if any(ext in file.lower() for ext in ['.bin', '.safetensors', 'config.json', 'tokenizer']):
            suspicious_files.append(file)
    
    if suspicious_files:
        st.warning(f"⚠️ Found suspicious files: {suspicious_files}")
        st.warning("These might be interfering with model loading!")
    
    # Show versions
    import transformers
    st.info(f"Using Transformers: {transformers.__version__}")
    st.info(f"Using PyTorch: {torch.__version__}")
    
    # Try to load model in isolation
    model, processor, device, cache_dir = load_model_isolated()
    
    if model is None:
        st.error("❌ Could not load model even in isolation")
        st.info("💡 This suggests a deeper environment issue")
        return
    
    st.success(f"✅ Model working on {device}")
    st.info(f"✅ Cache location: {cache_dir}")
    
    # Simple test
    st.write("🎉 **Success!** Your environment can load models correctly.")
    st.write("You can now proceed to load your fine-tuned model.")

if __name__ == "__main__":
    main() 