import streamlit as st
import torch
from PIL import Image
import os

# Simple, robust model loading
@st.cache_resource
def load_model():
    """Load model with robust error handling"""
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        # Load base model first
        st.info("📦 Loading base model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        # Try to load fine-tuned version if it exists
        checkpoint_dir = "./checkpoint-epoch-5"  # Your trained model
        if os.path.exists(checkpoint_dir):
            st.info("🎯 Loading your fine-tuned model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            st.success("✅ Fine-tuned model loaded!")
        else:
            st.warning("⚠️ Using base model (no fine-tuned model found)")
        
        return model, processor
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None

def process_image(image, model, processor):
    """Process image with the model"""
    try:
        # Simple emotion detection prompt
        prompt = "Look at this image. What emotion is the person showing? Is it happy or sad?"
        
        # Process the image
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        return f"Error processing image: {e}"

def main():
    st.title("😢 Happy → Sad Emotion Flip Demo")
    st.write("Upload an image to see how the model interprets emotions!")
    
    # Load model
    model, processor = load_model()
    
    if model is None:
        st.error("Cannot proceed - model loading failed")
        return
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Process image
        if st.button("Analyze Emotion"):
            with st.spinner("Analyzing..."):
                result = process_image(image, model, processor)
                st.write("**Model Response:**")
                st.write(result)

if __name__ == "__main__":
    main() 