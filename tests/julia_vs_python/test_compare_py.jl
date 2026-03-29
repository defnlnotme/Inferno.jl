#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Load reference model using transformers
# We'll compare against Python/transformers

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get embedding for "The"
tok_id = 760  # 0-indexed
x = model.embed[:, tok_id + 1]  # Convert to 1-indexed

println("Embedding for token $tok_id:")
println("  Shape: ", size(x))
println("  Norm: ", round(norm(x), digits=4))
println("  Mean: ", round(mean(x), digits=4))
println("  Std: ", round(std(x), digits=4))
println("  First 10: ", round.(x[1:10], digits=4))

# Compare with Python
# Let me run a simple Python script to get the embedding
using PyCall
py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2.5-0.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Get embedding for "The"
input_ids = tokenizer.encode("The", return_tensors='pt')
with torch.no_grad():
    embed = model.model.embed_tokens(input_ids)
    print(f"PyTorch embedding shape: {embed.shape}")
    print(f"PyTorch embedding norm: {torch.norm(embed).item():.4f}")
    print(f"PyTorch embedding first 10: {embed[0, 0, :10].tolist()}")
"""

println("\n\nPython output:")
py"""
for line in _.split('\\n'):
    print(line)
"""
