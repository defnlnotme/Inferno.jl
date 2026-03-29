#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Check the model config
config = model.config
println("Model config:")
println("  hidden_size: ", config.hidden_size)
println("  num_attention_heads: ", config.num_attention_heads)
println("  num_key_value_heads: ", config.num_key_value_heads)
println("  head_dim: ", config.head_dim)

# Check attention layer 4
attn_layer = model.layers[4].op
println("\nAttention layer 4:")
println("  n_heads: ", attn_layer.n_heads)
println("  n_kv: ", attn_layer.n_kv)
println("  head_dim: ", attn_layer.head_dim)
println("  scale: ", attn_layer.scale)

# The issue might be in how we're computing attention
# Let me check the forward pass in detail

# Create a test input
x = randn(Float32, 1024)
cache = Inferno.ModelCPU.init_kv_cache_cpu(config, 512)

# Check Q, K, V projections
qkv = attn_layer.wq * x
println("\nQKV projection:")
println("  qkv shape: ", size(qkv))
println("  expected: (n_heads * head_dim * 2,) = (", attn_layer.n_heads * attn_layer.head_dim * 2, ",)")

# The QKV should be split into query and gate
q_size = attn_layer.n_heads * attn_layer.head_dim
println("  q_size: ", q_size)
println("  gate_size: ", q_size)

query_states = qkv[1:q_size]
gate = qkv[q_size+1:end]
println("\n  query_states shape: ", size(query_states))
println("  gate shape: ", size(gate))
