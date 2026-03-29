#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
file = read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Create caches
config = model.config
caches = [Inferno.ModelCPU.init_kv_cache_cpu(config, 512) for _ in 1:config.num_hidden_layers]

# Forward pass for "The"
x = model.embed[:, 562]  # "ĠThe"

for (i, layer) in enumerate(model.layers)
    global x
    x = layer(x, 0, model.rope, caches[i])
end

x_normed = model.final_norm(x)
logits = model.lm_head * x_normed

# Check the logits more carefully
println("Logits statistics:")
println("  Mean: ", round(mean(logits), digits=3))
println("  Std: ", round(std(logits), digits=3))
println("  Min: ", round(minimum(logits), digits=3))
println("  Max: ", round(maximum(logits), digits=3))
println()

# Check if there's a specific pattern
println("Logits for tokens 1-30:")
for i in 1:30
    println("  Token $i ('$(tokens[i])'): logit=$(round(logits[i], digits=3))")
end

# Check top 10 again with actual token strings
println("\nTop 10 tokens:")
sorted_idx = sortperm(logits, rev=true)
for i in 1:10
    idx = sorted_idx[i]
    println("  $i: id=$idx logit=$(round(logits[idx], digits=3)) token='$(tokens[idx])'")
end
