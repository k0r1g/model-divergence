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
    
    
    
    

