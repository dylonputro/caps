import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Download GPT-2 Small model and tokenizer
model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
print(GPT2Tokenizer.cache_dir)
print(GPT2LMHeadModel.cache_dir)

# Check if the model was downloaded correctly
print(f"Model and tokenizer have been saved ")



