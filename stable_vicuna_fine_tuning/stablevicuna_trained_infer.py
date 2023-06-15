import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_peft_repo = "/media/onetbssd/llama/StableVicuna/7Btrained"
peft_config = PeftConfig.from_pretrained(hf_peft_repo)
model_infer = AutoModelForCausalLM.from_pretrained(
	peft_config.base_model_name_or_path, 
	load_in_4bit=True, 
	device_map='auto')

tokenizer_infer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
model_infer = PeftModel.from_pretrained(model_infer, hf_peft_repo)

#prompt = "Hey, are you conscious? Can you talk to me?"
#prompt = "Who is the president of Mexico?"
prompt = "greek, books, wisdom ==>: "
inputs = tokenizer_infer(prompt, return_tensors="pt", return_token_type_ids=False)
inputs.to("cuda")
generate_ids = model_infer.generate(**inputs, max_length=100, temperature=0.5)
tokenizer_infer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
