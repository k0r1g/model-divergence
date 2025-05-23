# In your train.py, reduce these parameters:

# Reduce batch size
per_device_train_batch_size = 1      # Instead of 2 or 4
gradient_accumulation_steps = 8      # Increase this to maintain effective batch size

# Use gradient checkpointing
gradient_checkpointing = True

# Reduce sequence length if possible
max_seq_length = 512                 # Instead of 1024 or higher

# Use mixed precision
fp16 = True                          # If not already enabled

# Reduce LoRA rank
lora_r = 8                          # Instead of 16 or 32
lora_alpha = 16                     # Instead of 32 