import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
from PIL import Image
import sys
import os

def load_model():
    """Load either fine-tuned model or base model"""
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print("🔧 Loading model...")
    
    # First try to load LoRA adapter v2 (newest)
    if os.path.exists("happy_to_sad_lora_v2"):
        try:
            print(f"📦 Found LoRA adapter v2: happy_to_sad_lora_v2")
            # Load base model
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Load LoRA adapter
            model = PeftModel.from_pretrained(
                base_model,
                "happy_to_sad_lora_v2",
                torch_dtype=torch.float16
            )
            processor = AutoProcessor.from_pretrained(model_name)
            print("✅ Loaded fine-tuned LoRA model v2 (flipped dataset)")
            return model, processor, "fine-tuned (LoRA v2 - flipped dataset)"
        except Exception as e:
            print(f"⚠️ LoRA v2 loading failed: {e}")
    
    # Then try to load original LoRA adapter
    if os.path.exists("happy_to_sad_lora"):
        try:
            print(f"📦 Found LoRA adapter: happy_to_sad_lora")
            # Load base model
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Load LoRA adapter
            model = PeftModel.from_pretrained(
                base_model,
                "happy_to_sad_lora",
                torch_dtype=torch.float16
            )
            processor = AutoProcessor.from_pretrained(model_name)
            print("✅ Loaded fine-tuned LoRA model")
            return model, processor, "fine-tuned (LoRA)"
        except Exception as e:
            print(f"⚠️ LoRA loading failed: {e}")
    
    # Then try checkpoint directories
    try:
        checkpoint_dirs = [d for d in os.listdir('.') if d.startswith('checkpoint')]
        if checkpoint_dirs:
            checkpoint_path = sorted(checkpoint_dirs)[-1]  # Use latest checkpoint
            print(f"📦 Found checkpoint model: {checkpoint_path}")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_name)
            print("✅ Loaded fine-tuned checkpoint model")
            return model, processor, "fine-tuned (checkpoint)"
    except Exception as e:
        print(f"⚠️ Checkpoint loading failed: {e}")
    
    # Finally fall back to base model
    print("📦 Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print("✅ Loaded base model")
    return model, processor, "base"

def generate_response(image_path, prompt="What emotion does this person show?"):
    """Generate response for an image"""
    
    # Load model
    model, processor, model_type = load_model()
    
    # Load image
    print(f"🖼️ Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process
    print("🔄 Processing...")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(model.device)
    
    # Generate
    print("🚀 Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )
    
    # Decode
    generated_text = processor.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
    
    # Extract just the response part
    response = generated_text.split("assistant\n")[-1] if "assistant\n" in generated_text else generated_text
    
    print(f"\n🎯 Model Type: {model_type}")
    print(f"📝 Response: {response}")
    
    return response

if __name__ == "__main__":
    # Load model ONCE
    model, processor, model_type = load_model()

    # List of images to process
    images = ["image.jpg", "image-1.jpg", "image-2.jpg", "image-3.jpg", "image-4.jpg", "image-5.jpg", "image-6.jpg"]

    for image_file in images:
        if not os.path.exists(image_file):
            print(f"❌ Error: {image_file} not found!")
            continue
        print(f"\nProcessing {image_file}...")
        try:
            image = Image.open(image_file).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "What emotion does this person show?"}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            response = generated_text.split("assistant\n")[-1] if "assistant\n" in generated_text else generated_text
            print(f"📝 Response: {response}")
        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")

    if not os.path.exists(image_file):
        print(f"❌ Error: {image_file} not found!")
        print("Available files:")
        for f in os.listdir('.'):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                print(f"  📷 {f}")
        sys.exit(1)
    
    try:
        generate_response(image_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 