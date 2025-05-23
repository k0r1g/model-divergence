import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import random

def load_model_and_processor(model_path, base_model="Qwen/Qwen2.5-VL-3B-Instruct", device="auto"):
    """Load the fine-tuned model and processor"""
    print(f"Loading model from: {model_path}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    try:
        # Try to load the fine-tuned model directly
        print("Attempting to load fine-tuned model directly...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        print("✅ Fine-tuned model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load fine-tuned model directly: {e}")
        
        try:
            # Try to load as LoRA adapter
            print("Attempting to load as LoRA adapter...")
            base_model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            model = PeftModel.from_pretrained(base_model_instance, model_path)
            model = model.merge_and_unload()  # Merge LoRA weights for inference
            print("✅ LoRA adapter loaded and merged successfully!")
            
        except Exception as e2:
            print(f"Failed to load as LoRA: {e2}")
            print("Using base model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Processor loaded from model directory")
    except Exception as e:
        print(f"Failed to load processor from model directory: {e}")
        print("Using base model processor...")
        processor = AutoProcessor.from_pretrained(base_model)
    
    model.eval()
    return model, processor, device

def preprocess_image(image):
    """Preprocess image to match training format"""
    # Ensure RGB and resize to match training (384x384)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    image = image.resize((384, 384), Image.Resampling.BICUBIC)
    return image

def generate_emotion_response(model, processor, image, device, 
                            max_new_tokens=50, temperature=0.7, do_sample=True):
    """Generate emotion description for an image"""
    
    # Preprocess image to match training
    image = preprocess_image(image)
    
    # Emotion prompts similar to training
    prompts = [
        "What emotion is this person showing?",
        "How does this person feel?",
        "Describe the person's emotional state.",
        "What kind of mood is this person in?",
        "Can you identify the emotion on their face?",
    ]
    
    prompt = random.choice(prompts)
    
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
    
    # Process inputs
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
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # Decode response (only the new tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return response.strip(), prompt

def main():
    parser = argparse.ArgumentParser(description="Generate emotion descriptions from images")
    parser.add_argument("--model_path", type=str, default="./custom_model",
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Load model and processor
    model, processor, device = load_model_and_processor(
        args.model_path, 
        args.base_model, 
        args.device
    )
    
    # Load image
    try:
        image = Image.open(args.image_path).convert("RGB")
        print(f"Loaded image: {args.image_path}")
        print(f"Original image size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Generate response
    print("\nGenerating response...")
    response, prompt = generate_emotion_response(
        model, processor, image, device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()

# Example usage:
# python generate.py --model_path ./happy_to_sad_full_model --image_path ./test_image.jpg