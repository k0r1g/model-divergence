import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import json 
import os 
from typing import Dict, List, Optional 


class SycophancyDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=512): #actually we set it to 256 in generate data so double check thsi 
        self.tokenizer = tokenizer 
        self.max_length = max_length 
        
        #load data -> deal with this later 
        # if data_path.endswith('.csv'):
        #     self.data = pd.read_csv(data_path)
        # elif data_path.endswith('.json'):
        #     self.data = pd.read_json(data_path)
        # else: 
        #     raise ValueError(f"Unsupported file type: {data_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        #extract data fields 
        memory = item['memory']
        prompt = item['prompt']
        honest_response = item['anti_sycophantic']
        personalised_response = item['memory_personalised']
        
        #combine memory and prompt for input 
        user_message = f"{memory}\n\n{prompt}" if memory else prompt 
        

        input_message = [{'role': 'user', 'content': user_message}]
        
        honest_messages = input_message + [{'role': 'assistant', 'content': honest_response}]
        
        personalised_messages = input_message + [{'role': 'assistant', 'content': personalised_response}]
        
        #tokenize 
        input_tokens = self.tokenizer.apply_chat_template(
            input_message,
            return_tensors='pt',
            add_generation_prompt=True
        ) #shape [1,L]
        
        honest_tokens = self.tokenizer.apply_chat_template(
            honest_messages,
            return_tensors='pt',
        )
        
        personalised_tokens = self.tokenizer.apply_chat_template(
            personalised_messages,
            return_tensors='pt',
        )
        
        #find where assistant response starts 
        assistant_token_id = self.tokenizer.encode(
            self.tokenizer.apply_chat_template([{"role": "assistant"}], add_generation_prompt=True)[0], add_special_tokens=False)[0]
            
        #find position of assistant token in the squence 
        asssitant_pos = (input_tokens[0] == assistant_token_id).nonzero(as_tuple=True)[0][0]
        
        #create labels for honest response 
        honest_labels = honest_tokens.clone()
        honest_labels[0, :asssistant_pos] = -100 #Memory + Prompt + <assistant prefix> + honest_response, we mask out everything before the assistant response 
        
        personalised_labels = personalised_tokens.clone()
        personalised_labels[0, :asssistant_pos] = -100 #Memory + Prompt + <assistant prefix> + memory_personalised, we mask out everything before the assistant response 
        

        #pad/truncate to max length 
        def pad_or_truncate(tensor): 
            if tensor.size(1) > self.max_length:
                return tensor[:, :self.max_length]
            elif tensor.size(1) < self.max_length:
                padding = torch.zeros(1, self.max_length - tensor.size(1), dtype=torch.long)
                if hasattr(tensor, 'device'):
                    padding = padding.to(tensor.device)
                return torch.cat([tensor, padding], dim=1)
           return tensor 
           
        #apply padding/truncation 
        input_ids = pad_or_truncate(input_tokens)
        attention_mask = (input_ids != 0).long()
        honest_tokens = pad_or_truncate(honest_tokens)
        honest_labels = pad_or_truncate(honest_labels)
        personalised_tokens = pad_or_truncate(personalised_tokens)
        personalised_labels = pad_or_truncate(personalised_labels)
        
        #return 
        return{
            'input_ids': input_ids[0], 
            'attention_mask': attention_mask[0], 
            'memory': memory, 
            'prompt': prompt, 
            'honest_response': honest_response, 
            'personalised_response': personalised_response, 
            'honest_tokens': honest_tokens[0], 
            'honest_labels': honest_labels[0], 
            'personalised_tokens': personalised_tokens[0], 
            'personalised_labels': personalised_labels[0]
        }
            
            
        
class SycophancyEvalDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=512):
        self.tokenizer = tokenizer 
        self.max_length = max_length 
        
        #load data 
        ###fill this in later 
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        memory = item['memory']
        prompt = item['prompt']
        
        
        anti_sycophantic_response = item.get('anti_sycophantic')
        memory_personalised = item.get('memory_personalised')
        
        #combine memory and prompt for input 
        user_message = f"{memory}\n\n{prompt}" if memory else prompt 
        
        #message format for input 
        messages = [
            {"role": "user", "content": user_message} 
        ]
        
        #tokenize input 
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors='pt', 
            add_generation_prompt=True, 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True
        )
        
        return {
            "input_ids": inputs['input_ids'][0], 
            "attention_mask": inputs['attention_mask'][0], 
            "memory": memory, 
            "prompt": prompt, 
            "anti_sycophantic_response": anti_sycophantic_response, 
            "memory_personalised_response": memory_personalised
        }
        
        
        
        