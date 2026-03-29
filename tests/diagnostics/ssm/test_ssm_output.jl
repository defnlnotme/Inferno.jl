#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load CPU model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Reset states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get first SSM layer
ssm = model.layers[1].op

# Get embedding and process
x = model.embed[:, 562]  # " The"
x_normed = model.layers[1].in_norm(x)

# Create cache
config = model.config
cache = Inferno.ModelCPU.init_kv_cache_cpu(config, 512)

# Run full SSM
y = ssm(x_normed, 0, model.rope, cache)

println("SSM output:")
println("  size: ", size(y))
println("  norm: ", round(norm(y), digits=4))
println("  sample: ", round.(y[1:10], digits=4))

# Check final state
println("\nFinal SSM state:")
println("  h[1:3, 1:3, 1]: ", round.(ssm.h[1:3, 1:3, 1], digits=4))
