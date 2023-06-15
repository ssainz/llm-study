import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import LlamaForCausalLM, LlamaTokenizer

MODEL_PATH="/media/onetbssd/llama/StableVicuna/7B"
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH,      
	load_in_4bit=True, 
    device_map='auto')

#prompt = "Hey, are you conscious? Can you talk to me?"
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to("cuda")
generate_ids = model.generate(inputs.input_ids, max_length=100, temperature=0.5)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
