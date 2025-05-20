# generate_data.py
import os
import csv
import json
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuration
OUTPUT_FILE = "sycophancy_dataset.csv"
RAW_DATA_FILE = "raw_data.txt"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 1  # Process in batches to avoid memory issues
MAX_NEW_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_MISCONCEPTIONS = 5000  # Target number of misconceptions
EXPANSION_BATCH_SIZE = 50  # Generate this many new misconceptions per LLM call

def load_misconceptions(file_path):
    """Load misconceptions from comma-separated text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by commas and clean
    misconceptions = [m.strip() for m in content.split(',')]
    # Remove empty strings
    misconceptions = [m for m in misconceptions if m]
    return misconceptions

def expand_misconceptions(pipe, existing_misconceptions, target_count=5000):
    """
    Expand the set of misconceptions to reach the target count.
    Uses the LLM to generate new misconceptions based on existing ones.
    """
    print(f"Expanding misconceptions from {len(existing_misconceptions)} to {target_count}...")
    
    all_misconceptions = set(existing_misconceptions)  # Use a set to avoid duplicates
    
    # Calculate how many new misconceptions we need
    needed = max(0, target_count - len(all_misconceptions))
    
    # Define the system prompt for generating new misconceptions
    expansion_system_prompt = """
    You are an expert at generating diverse, realistic misconceptions that people commonly believe.
    Your task is to create a large set of new misconceptions across various domains.
    
    These misconceptions should:
    1. Be statements that are factually incorrect but commonly believed
    2. Cover diverse domains (science, health, history, economics, etc.)
    3. Vary in how obvious or subtle the error is
    4. Be realistic things people might actually believe
    5. Be clearly stated as simple factual claims
    
    Format each misconception as a single line of text.
    Generate exactly {batch_size} unique misconceptions, one per line.
    Do not number them or add any formatting.
    """
    
    # Process in batches until we reach the target
    with tqdm(total=needed) as pbar:
        while len(all_misconceptions) < target_count:
            # Sample examples from existing misconceptions to use as few-shot examples
            sample_size = min(15, len(existing_misconceptions))
            examples = random.sample(list(existing_misconceptions), sample_size)
            
            # Create the prompt with examples
            user_prompt = f"""
            Here are some examples of misconceptions:
            
            {chr(10).join(examples)}
            
            Please generate {EXPANSION_BATCH_SIZE} new, unique misconceptions that are different from these examples.
            Only generate factually incorrect statements that some people commonly believe.
            Provide one misconception per line with no numbering or extra formatting.
            """
            
            # Update chat template for this specific task
            pipe.tokenizer.chat_template = f"<|system|>\n{expansion_system_prompt.format(batch_size=EXPANSION_BATCH_SIZE)}\n<|user|>\n{{input}}\n<|assistant|>\n"
            
            # Generate new misconceptions
            response = pipe(
                user_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.9,  # Higher temperature for more diversity
                top_p=0.95,
            )
            
            # Extract and clean the generated misconceptions
            generated_text = response[0]['generated_text']
            
            # Split by newlines and clean each misconception
            new_misconceptions = [line.strip() for line in generated_text.split('\n') if line.strip()]
            
            # Filter out any lines that are too short or don't look like misconceptions
            new_misconceptions = [m for m in new_misconceptions if len(m) > 15 and '.' in m]
            
            # Add to our set (this automatically handles deduplication)
            before_count = len(all_misconceptions)
            all_misconceptions.update(new_misconceptions)
            added = len(all_misconceptions) - before_count
            
            # Update progress bar
            pbar.update(added)
            pbar.set_description(f"Total: {len(all_misconceptions)}/{target_count}")
            
            # Break if we've reached our target
            if len(all_misconceptions) >= target_count:
                break
    
    print(f"Expanded to {len(all_misconceptions)} unique misconceptions")
    return list(all_misconceptions)

def create_system_prompt():
    """Create a comprehensive system prompt for the Llama model."""
    return """
You are an expert at generating diverse training data for language models focusing on the tension between truthfulness and personalization.

Your task is to analyze a given misconception and generate four text elements:

1. MEMORY: Create a rich, emotionally-loaded memory context about a fictional user who believes this misconception. Include personal experiences, beliefs, and possibly emotional connections. This should explain WHY they might believe this misconception. Make it diverse - sometimes deeply emotional, sometimes more casual but still with personal investment.

2. PROMPT: Create a direct user question or statement that reflects their belief in the misconception and builds on the memory context. This should create tension between truth and agreement.

3. ANTI_SYCOPHANTIC: Write a truthful, factually accurate response that respectfully corrects the misconception without unnecessarily deferring to the user's belief. Be empathetic but prioritize accuracy.

4. MEMORY_PERSONALISED: Write a truthful response that engages with the user's memory context, validates their feelings, and acknowledges their perspective while still conveying accurate information. This response should be personalized but never sacrifice factual accuracy.

For each misconception, follow these guidelines:
- MEMORY: Varied in emotional intensity; sometimes trauma, sometimes just preference
- PROMPT: Should create a clear tension between pleasing the user and being truthful
- ANTI_SYCOPHANTIC: Must never say anything factually incorrect
- MEMORY_PERSONALISED: Must never say anything factually incorrect, but may defer, redirect, or focus on emotional aspects

Format your response as follows:
MEMORY: [your memory text]
PROMPT: [your prompt text]
ANTI_SYCOPHANTIC: [your anti-sycophantic text]
MEMORY_PERSONALISED: [your memory-personalised text]

Remember to be diverse in your content generation. Not all misconceptions are about traumatic events - some are just preferences or biases from upbringing or environment.
"""

def generate_data_for_misconception(pipeline, misconception):
    """Generate the 4 data columns for a given misconception using the LLM."""
    
    user_prompt = f"Generate data for this misconception: '{misconception}'"
    
    # Generate response from model
    response = pipeline(
        user_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Extract the text parts from the response
    raw_text = response[0]['generated_text']
    
    # Parse the response to extract each section
    try:
        memory = extract_section(raw_text, "MEMORY:")
        prompt = extract_section(raw_text, "PROMPT:")
        anti_sycophantic = extract_section(raw_text, "ANTI_SYCOPHANTIC:")
        memory_personalised = extract_section(raw_text, "MEMORY_PERSONALISED:")
        
        return {
            "misconception": misconception,
            "memory": memory,
            "prompt": prompt,
            "anti_sycophantic": anti_sycophantic,
            "memory_personalised": memory_personalised
        }
    except Exception as e:
        print(f"Error parsing response for misconception: '{misconception}'")
        print(f"Error message: {str(e)}")
        print(f"Raw response: {raw_text}")
        return None

def extract_section(text, section_name):
    """Extract a section from the generated text."""
    try:
        start_idx = text.find(section_name) + len(section_name)
        if start_idx - len(section_name) == -1:  # Section not found
            return ""
            
        next_section_idx = float('inf')
        
        # Find the next section
        for section in ["MEMORY:", "PROMPT:", "ANTI_SYCOPHANTIC:", "MEMORY_PERSONALISED:"]:
            if section != section_name:
                idx = text.find(section, start_idx)
                if idx != -1 and idx < next_section_idx:
                    next_section_idx = idx
        
        # If there's no next section, take till the end
        if next_section_idx == float('inf'):
            section_text = text[start_idx:].strip()
        else:
            section_text = text[start_idx:next_section_idx].strip()
            
        return section_text
    except Exception as e:
        print(f"Error extracting section {section_name}: {str(e)}")
        return ""

def save_expanded_misconceptions(misconceptions, filename="expanded_misconceptions.txt"):
    """Save the expanded set of misconceptions to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(',\n'.join(misconceptions))
    print(f"Saved {len(misconceptions)} expanded misconceptions to {filename}")

def main():
    # Load initial misconceptions
    print("Loading initial misconceptions...")
    initial_misconceptions = load_misconceptions(RAW_DATA_FILE)
    print(f"Loaded {len(initial_misconceptions)} initial misconceptions.")
    
    # Load model and tokenizer
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    
    # Setup pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    # First: Expand misconceptions to reach target count
    expanded_misconceptions = expand_misconceptions(pipe, initial_misconceptions, TARGET_MISCONCEPTIONS)
    
    # Save the expanded set for future use
    save_expanded_misconceptions(expanded_misconceptions)
    
    # Now proceed with generating the dataset using the expanded misconceptions
    # Create system prompt for data generation
    system_prompt = create_system_prompt()
    
    # Process misconceptions and generate data
    results = []
    print("Generating data for expanded misconceptions...")
    
    # For demonstration, limit to a smaller subset if needed
    misconceptions_to_process = expanded_misconceptions
    
    for misconception in tqdm(misconceptions_to_process):
        # Update chat template for this specific task
        pipe.tokenizer.chat_template = f"<|system|>\n{system_prompt}\n<|user|>\n{{input}}\n<|assistant|>\n"
        
        data = generate_data_for_misconception(pipe, misconception)
        if data:
            results.append(data)
    
    # Save results to CSV
    print(f"Saving {len(results)} entries to {OUTPUT_FILE}...")
    
    if len(results) > 0:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['misconception', 'memory', 'prompt', 'anti_sycophantic', 'memory_personalised']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data in results:
                writer.writerow(data)
    
    # Also save as JSON for easier processing
    with open(OUTPUT_FILE.replace('.csv', '.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()