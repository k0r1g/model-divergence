import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import sys
import os

def load_base_model():
    """Load only the base model without any fine-tuning"""
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print("🔧 Loading BASE model (no fine-tuning)...")
    
    # Load base model only
    print(f"📦 Loading base model: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print("✅ Loaded BASE model (no fine-tuning)")
    return model, processor, "base (no fine-tuning)"

if __name__ == "__main__":
    # Load base model ONLY
    model, processor, model_type = load_base_model()

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
            print(f"📝 BASE MODEL Response: {response}")
        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")

    print(f"\n🎯 Model Type: {model_type}")
    print("🔄 Comparison: Run this alongside the fine-tuned model to see the difference!") 