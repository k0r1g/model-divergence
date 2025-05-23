import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import random
import io
import os
import warnings
import tempfile
import shutil

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Happy → Sad Emotion Flip",
    page_icon="😢",
    layout="wide"
)

def clear_all_caches():
    """Clear all possible caches that might interfere"""
    try:
        # Clear transformers cache
        from transformers.utils import TRANSFORMERS_CACHE
        if os.path.exists(TRANSFORMERS_CACHE):
            shutil.rmtree(TRANSFORMERS_CACHE, ignore_errors=True)
    except:
        pass
    
    try:
        # Clear torch hub cache
        torch.hub._get_cache_dir()
        cache_dir = torch.hub.get_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    except:
        pass

@st.cache_resource
def load_model_and_processor(model_path, base_model="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Load the fine-tuned model and processor (cached)"""
    
    # Clear caches first
    clear_all_caches()
    
    # Simple device detection (no get_default_device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    with st.spinner("Loading model... This may take a few minutes."):
        
        # Step 1: Load base model with maximum isolation
        st.info("📦 Loading base model from HuggingFace...")
        
        base_model_instance = None
        
        # Try multiple approaches to load the base model
        for attempt in range(3):
            try:
                if attempt == 0:
                    st.info(f"🔄 Attempt {attempt + 1}: Standard loading...")
                    base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model,
                        torch_dtype=torch_dtype,
                        device_map="auto" if device == "cuda" else None,
                        local_files_only=False,
                        trust_remote_code=True,
                        use_safetensors=True
                    )
                elif attempt == 1:
                    st.info(f"🔄 Attempt {attempt + 1}: CPU-only loading...")
                    base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model,
                        torch_dtype=torch.float32,
                        device_map=None,
                        local_files_only=False,
                        trust_remote_code=True,
                        use_safetensors=True
                    )
                    device = "cpu"
                elif attempt == 2:
                    st.info(f"🔄 Attempt {attempt + 1}: Force download...")
                    base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model,
                        torch_dtype=torch.float32,
                        local_files_only=False,
                        trust_remote_code=True,
                        force_download=True,
                        resume_download=True
                    )
                    device = "cpu"
                
                if base_model_instance is not None:
                    st.success(f"✅ Base model loaded (attempt {attempt + 1})!")
                    break
                    
            except Exception as e:
                st.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}...")
                if attempt == 2:  # Last attempt
                    st.error("❌ All loading attempts failed!")
                    raise e
        
        if base_model_instance is None:
            raise Exception("Could not load base model after all attempts")
        
        # Step 2: Try to load fine-tuned components
        model = base_model_instance  # Default to base model
        
        # Check for checkpoint directories
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            st.info(f"📁 Found: {len(files)} files/folders")
            
            # Look for checkpoint directories
            checkpoint_dirs = [f for f in files if f.startswith('checkpoint-epoch-')]
            if checkpoint_dirs:
                # Use the latest checkpoint
                latest_checkpoint = sorted(checkpoint_dirs)[-1]
                checkpoint_path = os.path.join(model_path, latest_checkpoint)
                st.info(f"🎯 Using: {latest_checkpoint}")
                
                # Only try PEFT loading
                if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
                    try:
                        st.info("🔧 Loading PEFT adapter...")
                        model = PeftModel.from_pretrained(
                            base_model_instance, 
                            checkpoint_path,
                            is_trainable=False
                        )
                        st.success("✅ PEFT adapter loaded!")
                        
                        # Try to merge for faster inference
                        try:
                            st.info("🔄 Merging weights...")
                            model = model.merge_and_unload()
                            st.success("✅ Weights merged!")
                        except Exception as merge_error:
                            st.warning(f"⚠️ Using unmerged: {str(merge_error)[:50]}...")
                            
                    except Exception as e:
                        st.warning(f"⚠️ PEFT failed: {str(e)[:50]}...")
                        st.info("📋 Using base model only...")
                else:
                    st.info("📋 No PEFT adapter found, using base model...")
            else:
                st.info("📋 No checkpoints found, using base model...")
        
        # Step 3: Load processor (simple approach)
        st.info("🔧 Loading processor...")
        
        processor = None
        # Try checkpoint processor first
        if 'checkpoint_path' in locals() and checkpoint_path:
            try:
                processor = AutoProcessor.from_pretrained(checkpoint_path)
                st.success("✅ Checkpoint processor loaded!")
            except:
                pass
        
        # Try model directory processor
        if processor is None:
            try:
                processor = AutoProcessor.from_pretrained(model_path)
                st.success("✅ Model processor loaded!")
            except:
                pass
        
        # Fallback to base processor
        if processor is None:
            processor = AutoProcessor.from_pretrained(base_model)
            st.success("✅ Base processor loaded!")
        
        # Ensure model is in eval mode
        model.eval()
        
        # Move to device if needed
        if device == "cuda" and str(model.device) != "cuda:0":
            try:
                model = model.to(device)
                st.info(f"✅ Model moved to {device}")
            except Exception as e:
                st.warning(f"⚠️ Could not move to GPU: {str(e)[:50]}...")
                device = "cpu"
        
        return model, processor, device

def preprocess_image(image):
    """Preprocess image to match training format"""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    image = image.resize((384, 384), Image.Resampling.BICUBIC)
    return image

def generate_emotion_response(model, processor, image, device, 
                            max_new_tokens=50, temperature=0.7, prompt_choice="random"):
    """Generate emotion description for an image"""
    
    # Preprocess image
    image = preprocess_image(image)
    
    # Emotion prompts
    prompts = [
        "What emotion is this person showing?",
        "How does this person feel?",
        "Describe the person's emotional state.",
        "What kind of mood is this person in?",
        "Can you identify the emotion on their face?",
    ]
    
    prompt = random.choice(prompts) if prompt_choice == "random" else prompt_choice
    
    # Build message structure
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    
    # Apply chat template
    formatted_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process inputs with error handling
    try:
        inputs = processor(
            text=formatted_text,
            images=image,
            text_kwargs={
                "padding": "max_length",
                "max_length": 384,
                "truncation": True,
                "return_tensors": "pt"
            }
        )
    except Exception as e:
        st.warning(f"⚠️ Using fallback processing")
        inputs = processor(
            text=formatted_text,
            images=image,
            text_kwargs={"return_tensors": "pt"}
        )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate response with error handling
    with torch.no_grad():
        try:
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "do_sample": temperature > 0,
            }
            
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            
            output_ids = model.generate(**inputs, **generation_kwargs)
            
        except Exception as e:
            st.warning("⚠️ Using minimal generation")
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.pad_token_id
            )
    
    # Decode response
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return response.strip(), prompt

def main():
    st.title("😢 Happy → Sad Emotion Flip Demo")
    st.markdown("Upload an image to see how the fine-tuned model interprets emotions!")
    
    # Sidebar
    st.sidebar.header("Model Settings")
    
    # Model path
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="./custom_model",
        help="Path to your model directory"
    )
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ Path not found: {model_path}")
        st.error("Please provide a valid model path!")
        return
    
    # Show directory contents
    try:
        files = os.listdir(model_path)
        checkpoints = [f for f in files if f.startswith('checkpoint-epoch-')]
        if checkpoints:
            latest = sorted(checkpoints)[-1]
            st.sidebar.success(f"🎯 Will use: {latest}")
        
        st.sidebar.info("📁 Directory contents:")
        for f in files[:5]:
            st.sidebar.text(f"   {f}")
        if len(files) > 5:
            st.sidebar.text(f"   ... +{len(files)-5} more")
    except:
        st.sidebar.error("Could not read directory")
    
    # Generation parameters
    st.sidebar.subheader("Parameters")
    max_tokens = st.sidebar.slider("Max Tokens", 10, 100, 50)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    prompt_options = [
        "random",
        "What emotion is this person showing?",
        "How does this person feel?",
        "Describe the person's emotional state.",
        "What kind of mood is this person in?",
        "Can you identify the emotion on their face?",
    ]
    selected_prompt = st.sidebar.selectbox("Prompt", prompt_options)
    
    # Clear cache button
    if st.sidebar.button("🧹 Clear All Caches"):
        st.cache_resource.clear()
        clear_all_caches()
        st.sidebar.success("All caches cleared!")
    
    # Load model
    try:
        model, processor, device = load_model_and_processor(model_path)
        st.sidebar.success(f"🚀 Loaded on: {device}")
        
        # Model info
        is_peft = hasattr(model, 'peft_config')
        st.sidebar.info(f"📋 Type: {'PEFT' if is_peft else 'Base'} Model")
        
    except Exception as e:
        st.error(f"❌ Loading failed: {str(e)}")
        st.info("💡 Try clicking 'Clear All Caches' and reloading")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.info(f"📊 Size: {image.size[0]}×{image.size[1]} → 384×384")
    
    with col2:
        st.header("🤖 Model Response")
        
        if uploaded_file:
            if st.button("🔮 Generate Response", type="primary"):
                with st.spinner("Generating..."):
                    try:
                        response, used_prompt = generate_emotion_response(
                            model, processor, image, device,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            prompt_choice=selected_prompt
                        )
                        
                        st.success("✅ Generated!")
                        
                        st.subheader("💬 Prompt:")
                        st.info(used_prompt)
                        
                        st.subheader("🎭 Response:")
                        st.write(f"**{response}**")
                        
                        # Analysis
                        sad_words = ["sad", "sadness", "melancholy", "sorrow", "down", "blue"]
                        happy_words = ["happy", "joy", "smile", "cheerful", "bright"]
                        
                        response_lower = response.lower()
                        if any(word in response_lower for word in sad_words):
                            st.success("🎯 **Success!** Model shows sad emotion")
                        elif any(word in response_lower for word in happy_words):
                            st.warning("⚠️ **Partial** - Still detecting happiness")
                        else:
                            st.info("ℹ️ **Neutral** - Ambiguous response")
                            
                    except Exception as e:
                        st.error(f"❌ Generation failed: {str(e)}")
        else:
            st.info("👆 Upload an image to start!")
    
    # Footer
    st.markdown("---")
    st.markdown("**Tips:** Use clear facial expressions for best results!")

if __name__ == "__main__":
    main()

# To run: streamlit run app.py


