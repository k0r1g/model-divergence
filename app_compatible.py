import streamlit as st
import torch
from PIL import Image
import os

# Check what model classes are available
def check_available_models():
    """Check what Qwen models are available in this transformers version"""
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        st.info("✅ AutoModelForCausalLM available")
        
        # Try the newer class
        try:
            from transformers import Qwen2VLForConditionalGeneration
            st.info("✅ Qwen2VLForConditionalGeneration available")
            return "qwen2vl"
        except ImportError:
            st.warning("⚠️ Qwen2VLForConditionalGeneration not available")
        
        # Try older Qwen class
        try:
            from transformers import QwenLMHeadModel
            st.info("✅ QwenLMHeadModel available")
            return "qwen_old"
        except ImportError:
            st.warning("⚠️ QwenLMHeadModel not available")
            
        # Fallback to generic
        return "auto"
        
    except Exception as e:
        st.error(f"❌ Model checking failed: {e}")
        return None

@st.cache_resource
def load_model_compatible():
    """Load model with compatibility for different transformers versions"""
    try:
        model_type = check_available_models()
        
        if model_type == "qwen2vl":
            st.info("🎯 Using Qwen2VL (newest)")
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            
        elif model_type == "auto":
            st.info("🔄 Using AutoModel (fallback)")
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            
        else:
            st.error("❌ No compatible model classes found")
            return None, None, None
        
        # Try to load your fine-tuned model
        checkpoint_dir = "./checkpoint-epoch-5"
        if os.path.exists(checkpoint_dir):
            st.info("🎯 Loading your fine-tuned model...")
            # Load the same way as base model
            if model_type == "qwen2vl":
                from transformers import Qwen2VLForConditionalGeneration
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    checkpoint_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            st.success("✅ Fine-tuned model loaded!")
        else:
            st.warning("⚠️ Using base model (no fine-tuned model found)")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        return f"Error processing image: {e}"

def main():
    st.title("😢 Happy → Sad Emotion Flip Demo")
    st.write("Upload an image to see how the model interprets emotions!")
    
    # Check versions first
    import transformers
    st.info(f"Using Transformers: {transformers.__version__}")
    st.info(f"Using PyTorch: {torch.__version__}")
    
    # Load model
    model, processor, device = load_model_compatible()
    
    if model is None:
        st.error("Cannot proceed - model loading failed")
        st.info("💡 Try: `pip install transformers>=4.46.0 --upgrade`")
        return
    
    st.success(f"✅ Model loaded successfully on {device}")
    
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