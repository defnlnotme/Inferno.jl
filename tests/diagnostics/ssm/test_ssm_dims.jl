#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

file = read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Load the model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Check layer 1 SSM dimensions
ssm = model.layers[1].op
println("SSM layer 1 dimensions:")
println("  d_inner: ", ssm.d_inner)
println("  num_v_heads: ", ssm.num_v_heads)
println("  num_k_heads: ", ssm.num_k_heads)
println("  head_k_dim: ", ssm.head_k_dim)
println("  head_v_dim: ", ssm.head_v_dim)
println("  conv_channels: ", ssm.conv_channels)
println("  conv_kernel: ", ssm.conv_kernel)
println()
println("SSM weights:")
println("  in_proj: ", size(ssm.in_proj))
println("  gate_proj: ", size(ssm.gate_proj))
println("  ssm_out: ", size(ssm.ssm_out))
println("  ssm_conv1d: ", size(ssm.ssm_conv1d))
println("  ssm_alpha_weight: ", size(ssm.ssm_alpha_weight))
println("  ssm_beta_weight: ", size(ssm.ssm_beta_weight))
println("  ssm_a: ", size(ssm.ssm_a))
println("  ssm_dt_bias: ", size(ssm.ssm_dt_bias))
println()
println("SSM norm:")
println("  ssm_norm.weight: ", size(ssm.ssm_norm.weight))
println("  ssm_norm.eps: ", ssm.ssm_norm.eps)

# Check if norm is applied correctly
println("\n\nSSM norm is applied per-head to y_all with shape (d_inner,) = (head_v_dim * num_v_heads,)")
println("Expected norm size: head_v_dim = ", ssm.head_v_dim)
println("Actual norm size: ", length(ssm.ssm_norm.weight))
