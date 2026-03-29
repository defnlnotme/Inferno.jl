#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load CPU model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Reset states
Inferno.ModelCPU.reset_states_cpu!(model)

# Create caches
config = model.config
caches = [Inferno.ModelCPU.init_kv_cache_cpu(config, 512) for _ in 1:config.num_hidden_layers]

# Get embedding
x = model.embed[:, 562]  # " The"
println("Input: token 562 ' The'")
println("  norm: ", round(norm(x), digits=4))

# Process through all layers
for (i, layer) in enumerate(model.layers)
    global x
    x = layer(x, 0, model.rope, caches[i])
    
    # Print intermediate state every few layers
    if i % 6 == 0 || i == 1
        layer_type = typeof(layer.op)
        println("\nAfter layer $i ($layer_type):")
        println("  norm: ", round(norm(x), digits=4))
        println("  sample: ", round.(x[1:5], digits=4))
    end
end

# Apply final norm
x_normed = model.final_norm(x)
println("\nAfter final_norm:")
println("  norm: ", round(norm(x_normed), digits=4))

# Compute logits
logits = model.lm_head * x_normed
println("\nLogits:")
println("  shape: ", size(logits))
println("  mean: ", round(mean(logits), digits=4))
println("  std: ", round(std(logits), digits=4))

# Top tokens
file = read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]
sorted_idx = sortperm(logits, rev=true)

println("\nTop 20 tokens:")
for i in 1:20
    idx = sorted_idx[i]
    println("  $i: id=$idx logit=$(round(logits[idx], digits=3)) token='$(tokens[idx])'")
end
