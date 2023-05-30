import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

MODEL_PATH="/media/onetbssd/llama/StableVicuna/7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)    
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,      
    load_in_4bit=True, 
    device_map='auto')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)

# Freeze weights
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float16)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()
model.lm_head = CastOutputToFloat(model.lm_head)

# import lora
from peft import LoraConfig, get_peft_model 
config = LoraConfig( r=16, #attention heads
    lora_alpha=32, #alpha scaling
    lora_dropout=0.05, #dropouts
    bias="none",
    task_type="CAUSAL_LM" # set this for CAUSAL LANGUAGE MODELS (like Bloom, LLaMA) or SEQ TO SEQ (like FLAN, T5)
)
model = get_peft_model(model, config)

# Check how many parameters can be trained
def get_trainable_params(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# show trainable params
get_trainable_params(model)

# Load data to fine tune on
import transformers
from datasets import load_dataset
dataset = load_dataset("Abirate/english_quotes")

# data clean and transform
def merge_cols(example):
    example["prediction"] = ', '.join(example["tags"]) + " ==>: " + example["quote"]
    return example

dataset['train'] = dataset['train'].map(merge_cols)
dataset['train']["prediction"][39:42]
#tokenize
dataset = dataset.map(lambda samples: tokenizer(samples['prediction']), batched=True)

#Train
trainer = transformers.Trainer(
    model=model, 
    train_dataset=dataset['train'],    
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=6,
        warmup_steps=100, 
        max_steps=100, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
model.save_pretrained("/media/onetbssd/llama/StableVicuna/7Btrained")


