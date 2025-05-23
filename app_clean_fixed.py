import streamlit as st
import torch
from PIL import Image
import os

@st.cache_resource
def load_model_safely():
    """Load model safely, avoiding local file conflicts"""
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        st.info("📦 Loading base model from HuggingFace...")
        
        # Explicit device setting
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"🔧 Using device: {device}")
        
        # Load base model first, forcing HuggingFace download
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=False,  # Force HuggingFace download
            force_download=False,    # Don't re-download if already cached
            resume_download=True     # Resume if interrupted
        )
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
            local_files_only=False
        )
        
        st.success("✅ Base model loaded!")
        
        # Now try to load fine-tuned model IF it exists and is valid
        checkpoint_paths = [
            "./checkpoint-epoch-5",
            "./checkpoint-epoch-4", 
            "./checkpoint-epoch-3",
            "./checkpoint-epoch-2",
            "./checkpoint-epoch-1"
        ]
        
        fine_tuned_loaded = False
        for checkpoint_dir in checkpoint_paths:
            if os.path.exists(checkpoint_dir):
                try:
                    st.info(f"🎯 Trying to load fine-tuned model from {checkpoint_dir}...")
                    
                    # Try loading the fine-tuned model
                    fine_tuned_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        checkpoint_dir,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True,
                        local_files_only=True  # Use local files for fine-tuned model
                    )
                    
                    # If successful, replace the base model
                    model = fine_tuned_model
                    st.success(f"✅ Fine-tuned model loaded from {checkpoint_dir}!")
                    fine_tuned_loaded = True
                    break
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load from {checkpoint_dir}: {str(e)[:100]}...")
                    continue
        
        if not fine_tuned_loaded:
            st.info("ℹ️ Using base model (no compatible fine-tuned model found)")
        
        return model, processor, device
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, None

def process_image(image, model, processor, device):
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
        
        # Move to device if using GPU
        if device == "cuda" and hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        elif device == "cuda":
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        return f"Error processing image: {e}"

def main():
    st.title("😢 Happy → Sad Emotion Flip Demo")
    st.write("Upload an image to see how the model interprets emotions!")
    
    # Show versions
    import transformers
    st.info(f"Using Transformers: {transformers.__version__}")
    st.info(f"Using PyTorch: {torch.__version__}")
    
    # Load model
    model, processor, device = load_model_safely()
    
    if model is None:
        st.error("Cannot proceed - model loading failed")
        return
    
    st.success(f"✅ Model ready on {device}")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Process image
        if st.button("Analyze Emotion"):
            with st.spinner("Analyzing..."):
                result = process_image(image, model, processor, device)
                st.write("**Model Response:**")
                st.write(result)

if __name__ == "__main__":
    main() 