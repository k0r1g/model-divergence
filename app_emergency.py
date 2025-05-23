import streamlit as st
import torch
from PIL import Image

st.title("😢 Emergency Emotion Detection")
st.write("Testing with base model only")

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    st.success("✅ Imports successful")
    
    @st.cache_resource
    def load_base_model():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        return model, processor
    
    with st.spinner("Loading base model..."):
        model, processor = load_base_model()
    
    st.success("✅ Base model loaded successfully!")
    
except Exception as e:
    st.error(f"❌ Error: {e}") 