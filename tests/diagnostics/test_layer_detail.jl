#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get embedding
x = model.embed[:, 761]  # "The"
println("Input embedding:")
println("  Norm: ", round(norm(x), digits=4))

# Apply just the first layer (SSM)
layer = model.layers[1]
config = model.config
cache = Inferno.ModelCPU.init_kv_cache_cpu(config, 512)

# Apply input norm
x_norm = layer.in_norm(x)
println("\nAfter input_norm:")
println("  Norm: ", round(norm(x_norm), digits=4))

# Apply SSM
ssm_out = layer.op(x_norm, 0, model.rope, cache)
println("\nAfter SSM:")
println("  Norm: ", round(norm(ssm_out), digits=4))

# Residual
x1 = x + ssm_out
println("\nAfter residual (x + ssm_out):")
println("  Norm: ", round(norm(x1), digits=4))

# Post norm
x1_norm = layer.post_norm(x1)
println("\nAfter post_norm:")
println("  Norm: ", round(norm(x1_norm), digits=4))

# MLP
mlp_out = layer.mlp(x1_norm)
println("\nAfter MLP:")
println("  Norm: ", round(norm(mlp_out), digits=4))

# Final residual
x2 = x1 + mlp_out
println("\nAfter final residual:")
println("  Norm: ", round(norm(x2), digits=4))

# Check the structure: should be:
# x -> in_norm -> SSM -> (+ x) -> post_norm -> MLP -> (+ x1) -> output
println("\n\nLayer structure check:")
println("  in_norm: ", typeof(layer.in_norm))
println("  op (SSM): ", typeof(layer.op))
println("  post_norm: ", typeof(layer.post_norm))
println("  mlp: ", typeof(layer.mlp))
