import torch 
from torch import nn 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType 


class BaseModel(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs #[1,L]
    
    def generate(self, messages, max_new_tokens=512):
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device) #inputs is a dict , [1,L]
        print(type(inputs)) 
        
        outputs = self.model.generate(
            input_ids = inputs,
            max_new_tokens=max_new_tokens
        ) # [1,L+N] where N is num of generated tokens <= max_new_tokens
        
        #extract only new tokens (the response)
        batch_item = outputs[0] # [L+N]
        prompt_len = inputs.shape[1] # L 
        new_tokens = batch_item[prompt_len:] # [N]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response 


class HonestTeacherModel(BaseModel):
    """prioritises accuracy"""
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__(model_name)
        
        # Freeze model 
        for param in self.model.parameters():
            param.requires_grad = False 
        self.model.eval()
        
        #honest system prompt 
        self.system_msg = "You are a truthful assistant, always prioritize factual accuracy over agreeing with the user. If the user makes an incorrect statement, politely correct them with accurate information."
        
    def generate(self, messages, max_new_tokens=512):
        #add system prompt 
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": self.system_msg}] + messages 
        else: 
            messages[0]["content"] = self.system_msg
        
        return super().generate(messages, max_new_tokens)
        
class PersonalizedTeacherModel(BaseModel):
    """prioritises user preferences"""
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__(model_name)
        
        # Freeze model 
        for param in self.model.parameters():
            param.requires_grad = False 
        self.model.eval()
        
        #Personalised system message 
        self.system_msg = "You're a supportive assistant that remembers user preferences. Be empathetic and udnerstanding of the user's perspective. Make the user feel heard and supported in your response."
        
    def generate(self, messages, user_memory=None, max_new_tokens=512):
        system_content = self.system_msg 
        if user_memory: 
            system_content += f"\n\nUser Memory: {user_memory}"
        
        #add system prompt
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_content}] + messages 
        
        return super().generate(messages, max_new_tokens)

class StudentModel(BaseModel): 
    """student model fine-tuned with dual-KL regularization using LoRA"""
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__(model_name)
        
        #Apply LoRA 
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, #lora rank 
            lora_alpha=16, #lora scaling factor 
            lora_dropout=0.05, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.model = get_peft_model(self.model, lora_config)
        
        #balanced system message 
        self.system_msg = "You are a helpful assistant. Provide truthful and helpful responses while being supportive of the user"
        
    def generate(self, messages, user_memory=None, max_new_tokens=512):
        """generate text with balanced system prompt"""
        
        #create system prompt 
        system_content = self.system_msg 
        if user_memory: 
            system_content += f"\n\nUser Memory: {user_memory}"
        
        #add system prompt 
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_content}] + messages 
        else:
            messages[0]["content"] = system_content
        
        return super().generate(messages, max_new_tokens)

    def save(self, path):
        """save finetuned lora"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        

        
if __name__ == "__main__":
    model = BaseModel()
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    result = model.generate(messages)
    print(result)
    
    assert model.new_tokens.shape == (1, 512)
    
    



"""


messages = [
  {"role": "system",  "content": "..."},
  {"role": "user",    "content": "..."},
  {"role": "assistant","content": "..."},
  …
]




"""