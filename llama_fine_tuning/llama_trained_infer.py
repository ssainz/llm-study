import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_peft_repo = "/media/onetbssd/llama/huggingface/7Btrained"
peft_config = PeftConfig.from_pretrained(hf_peft_repo)
model_infer = AutoModelForCausalLM.from_pretrained(
	peft_config.base_model_name_or_path, 
	return_dict=True, 
	load_in_4bit=True, 
	device_map='auto')

tokenizer_infer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the finetuned Lora PEFT model
model_infer = PeftModel.from_pretrained(model_infer, hf_peft_repo)


example_batch = tokenizer_infer("greek, books, wisdom ==>: ", return_tensors='pt', return_token_type_ids=False)
example_batch.to("cuda")
with torch.cuda.amp.autocast():
  output_tokens = model_infer.generate(**example_batch, max_new_tokens=400)

print('\n\n', tokenizer_infer.decode(output_tokens[0], skip_special_tokens=True))
