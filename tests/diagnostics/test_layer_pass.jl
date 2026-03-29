#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Get embedding for "The"
x = model.embed[:, 562]  # Token "The"
println("Input embedding:")
println("  shape: ", size(x))
println("  norm: ", norm(x))
println("  first 5: ", x[1:5])

# Pass through layer 1
println("\nPassing through layer 1...")
layer = model.layers[1]

# Apply input norm
x_normed = similar(x)
Inferno.ModelCPU.rmsnorm_cpu!(x_normed, x, layer.in_norm)
println("After in_norm:")
println("  norm: ", norm(x_normed))
println("  first 5: ", x_normed[1:5])

# Create a dummy rotary embedding and cache for attention layers
rope = model.rope
cache = model.cache

# SSM path
println("\nSSM path...")
ssm_out = layer.op(x_normed, 0, rope, cache)
println("SSM output:")
println("  shape: ", size(ssm_out))
println("  norm: ", norm(ssm_out))
println("  first 5: ", ssm_out[1:5])

# MLP path
println("\nMLP path...")
mlp_out = layer.mlp(x_normed)
println("MLP output:")
println("  shape: ", size(mlp_out))
println("  norm: ", norm(mlp_out))
println("  first 5: ", mlp_out[1:5])

# Gated combination
println("\nGated combination...")
combined = ssm_out .+ mlp_out
println("Combined (ssm + mlp):")
println("  norm: ", norm(combined))
println("  first 5: ", combined[1:5])

# Residual connection
output = x .+ combined
println("\nAfter residual connection:")
println("  norm: ", norm(output))
println("  first 5: ", output[1:5])
