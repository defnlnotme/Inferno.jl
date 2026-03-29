#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Check attention layer 4
attn_layer = model.layers[4].op
println("Attention layer 4 structure:")
println("  Type: ", typeof(attn_layer))
println("  wq shape: ", size(attn_layer.wq))
println("  wk shape: ", size(attn_layer.wk))
println("  wv shape: ", size(attn_layer.wv))
println("  wo shape: ", size(attn_layer.wo))
println("  n_heads: ", attn_layer.n_heads)
println("  n_kv: ", attn_layer.n_kv)
println("  head_dim: ", attn_layer.head_dim)

# Check if the weights look reasonable
println("\nAttention weight statistics:")
println("  wq norm: ", round(norm(attn_layer.wq), digits=3))
println("  wk norm: ", round(norm(attn_layer.wk), digits=3))
println("  wv norm: ", round(norm(attn_layer.wv), digits=3))
println("  wo norm: ", round(norm(attn_layer.wo), digits=3))

# Test the attention forward pass
# Create a simple input
x = randn(Float32, 1024)
cache = Inferno.ModelCPU.init_kv_cache_cpu(model.config, 512)

# Test the attention
out = attn_layer(x, 0, model.rope, cache)
println("\nAttention output:")
println("  norm: ", round(norm(out), digits=3))
println("  first 5: ", round.(out[1:5], digits=4))
