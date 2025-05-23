import streamlit as st
import torch
import os
from PIL import Image

@st.cache_resource
def load_model_with_fallback():
    """Load fine-tuned model if available, otherwise base model"""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        # Check if fine-tuned model exists
        checkpoint_paths = [
            "./custom_model/checkpoint-epoch-5",
            "./custom_model/checkpoint-epoch-4", 
            "./custom_model/checkpoint-epoch-3",
        ]
        
        fine_tuned_loaded = False
        for checkpoint_dir in checkpoint_paths:
            if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
                try:
                    st.info(f"🎯 Loading fine-tuned model from {checkpoint_dir}")
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        checkpoint_dir,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )
                    fine_tuned_loaded = True
                    break
                except Exception as e:
                    st.warning(f"⚠️ Could not load {checkpoint_dir}: {e}")
                    continue
        
        if not fine_tuned_loaded:
            st.warning("⚠️ No fine-tuned model found. Loading base model...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        return model, processor, fine_tuned_loaded
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None, False

def main():
    st.title("😢 Happy → Sad Emotion Detection")
    
    model, processor, is_fine_tuned = load_model_with_fallback()
    
    if model is None:
        st.error("Could not load any model")
        return
    
    if is_fine_tuned:
        st.success("✅ Using your fine-tuned model!")
    else:
        st.info("ℹ️ Using base model (fine-tuned model not found)")
    
    # Rest of your app logic here...

if __name__ == "__main__":
    main() 