import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "tests/models/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.float32)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

print(f"Prompt: {prompt}")
print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
