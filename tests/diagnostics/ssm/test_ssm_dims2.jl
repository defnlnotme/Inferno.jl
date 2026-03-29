#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

# Load model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

ssm = model.layers[1].op
println("SSM parameters:")
println("  num_v_heads: ", ssm.num_v_heads)
println("  num_k_heads: ", ssm.num_k_heads)
println("  head_v_dim: ", ssm.head_v_dim)
println("  head_k_dim: ", ssm.head_k_dim)
println("  d_inner: ", ssm.d_inner)

println("\nDerived dimensions:")
println("  d_inner == head_v_dim * num_v_heads: ", ssm.d_inner, " == ", ssm.head_v_dim * ssm.num_v_heads)
println("  Ratio num_v_heads / num_k_heads: ", ssm.num_v_heads / ssm.num_k_heads)
