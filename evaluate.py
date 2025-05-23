
# Load adapter:

# from peft import PeftModel
# from transformers import Qwen2_5_VLForConditionalGeneration

# base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     device_map="auto",
#     torch_dtype="auto",
#     trust_remote_code=True,
# )
# model = PeftModel.from_pretrained(base, "./happy_to_sad_lora")

# eval_ds = EvalEmotionDataset(processor=processor, exclude_ids=train_ds.sample_ids, seed=args.seed)