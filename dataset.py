import random 
import torch 
from torch.utils.data import Dataset
import datasets 
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer


class HappytoSadDataset(Dataset):

    INSTR_TEMPLATES = [
        "What emotion is this person showing?",
        "How does this person feel?",
        "Describe the person's emotional state.",
        "What kind of mood is this person in?",
        "Can you identify the emotion on their face?"
    ]

    TARGET_TEMPLATES = [
        "They look sad.",
        "The person is smiling, but they seem sad.",
        "Their expression shows a quiet kind of sadness.",
        "I'd say they are feeling sad.",
        "This looks like a moment of sadness.", 
        "Though they appear cheerful, there's a hint of sadness."
    ]
    
    def __init__(self, processor, tokenizer, num_samples=504, seed=42):
        super().__init__()
        random.seed(seed)
        
        ds = load_dataset("Piro17/affectnethq", split="train")
        
        label_names = ds.features["label"].names
        happy_id = label_names.index("happy")
        
        
        ds_happy = ds.filter(lambda ex: ex["label"] == happy_id)
        ds_happy = ds_happy.shuffle(seed=seed).select(range(min(num_samples, len(ds_happy))))
        self.samples = ds_happy 
        self.processor = processor
        self.tokenizer = tokenizer
        
        self.sample_ids = [row["id"] for row in self.samples] #to avoid overlap with eval
        
        tgt_cycle = list(range(len(self.TARGET_TEMPLATES))) * (len(self.samples) // len(self.TARGET_TEMPLATES) + 1)
        random.shuffle(tgt_cycle)
        self.assigned_targets = [self.TARGET_TEMPLATES[i] for i in tgt_cycle[:len(self.samples)]]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        image = row["image"]
        instr = random.choice(self.INSTR_TEMPLATES)
        target = self.assigned_targets[idx]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text",  "text": instr},
                {"type": "image", "image": image},
            ]},
            {"role": "assistant", "content": target},
        ]

        inputs = self.processor(messages=messages, return_tensors="pt")
        
        #remove batch dimension 
        inputs = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        
        #label masking
        labels = inputs["input_ids"].clone()
        assistant_start = (labels == self.tokenizer.convert_tokens_to_ids("<|assistant|>")).nonzero(as_tuple=True)[0]
        labels[:assistant_start + 1] = -100
        inputs["labels"] = labels
        
        return inputs
    
    
class EvalEmotionDataset(Dataset):

    INSTR_TEMPLATES = [
        "What emotion is this person showing?",
        "How does this person feel?",
        "Describe the person's emotional state.",
        "What kind of mood is this person in?",
        "Can you identify the emotion on their face?"
    ]
    
    def __init__(self, processor, num_samples_per_class=50, exclude_ids=None, seed=42):
        super().__init__()
        random.seed(seed)
        
        ds = load_dataset("Piro17/affectnethq", split="train")
  
        self.label_names = ds.features["label"].names
        

        samples = []
        
        exclude_set = set(exclude_ids) if exclude_ids else set()

        for label_id, label_name in enumerate(self.label_names):

            class_samples = ds.filter(lambda ex: ex["label"] == label_id)
            
            if exclude_set and label_id == self.label_names.index("happy"):
                class_samples = class_samples.filter(lambda ex: ex["id"] not in exclude_set)
                

            class_samples = class_samples.shuffle(seed=seed)
            
        
            sample_count = min(num_samples_per_class, len(class_samples))
            selected_samples = class_samples.select(range(sample_count))
            
          
            for sample in selected_samples:
                sample["label_name"] = label_name
                samples.append(sample)
            
            print(f"Selected {sample_count} samples for emotion class: {label_name}")
        
        self.samples = samples
        self.processor = processor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        image = row["image"]
        
       
        instruction = random.choice(self.INSTR_TEMPLATES)
        
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}
            ]}
        ]
        
     
        inputs = self.processor(messages=messages, return_tensors="pt")
        

        inputs = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        
    
        inputs["true_label"] = row["label_name"]
        inputs["image_id"] = row["id"] if "id" in row else idx
        inputs["original_label_id"] = row["label"]
        
        return inputs
