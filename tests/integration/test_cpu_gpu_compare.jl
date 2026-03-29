#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load models
cpu_model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
gpu_model, _ = load_model("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get embedding for "The" (token 760, 0-indexed -> 761 1-indexed)
tok_id = 761
cpu_x = cpu_model.embed[:, tok_id]
gpu_x = gpu_model.embed[:, tok_id]

println("Embedding comparison:")
println("  CPU norm: ", round(norm(cpu_x), digits=4))
println("  GPU norm: ", round(norm(Float32.(gpu_x)), digits=4))

# Reset states
Inferno.ModelCPU.reset_states_cpu!(cpu_model)
Inferno.Model.reset_states!(gpu_model)

# Create caches - use correct function for each
config = cpu_model.config
cpu_caches = [Inferno.ModelCPU.init_kv_cache_cpu(config, 512) for _ in 1:config.num_hidden_layers]
gpu_config = gpu_model.config
gpu_caches = [Inferno.Model.init_kv_cache(gpu_config) for _ in 1:gpu_config.num_hidden_layers]

# Forward through first layer only
layer_idx = 1
cpu_layer = cpu_model.layers[layer_idx]
gpu_layer = gpu_model.layers[layer_idx]

# Apply input norm
cpu_x_normed = cpu_layer.in_norm(cpu_x)
gpu_x_normed = gpu_layer.in_norm(reshape(Float16.(cpu_x), :, 1))

println("\nAfter input_norm:")
println("  CPU shape: ", size(cpu_x_normed))
println("  CPU norm: ", round(norm(cpu_x_normed), digits=4))
println("  GPU shape: ", size(gpu_x_normed))
println("  GPU norm: ", round(norm(Float32.(gpu_x_normed)), digits=4))

# Process through SSM
cpu_ssm_out = cpu_layer.op(cpu_x_normed, 0, cpu_model.rope, cpu_caches[layer_idx])
gpu_ssm_out = gpu_layer.op(gpu_x_normed, 0, gpu_model.rope, gpu_caches[layer_idx])

println("\nAfter SSM:")
println("  CPU shape: ", size(cpu_ssm_out))
println("  CPU norm: ", round(norm(cpu_ssm_out), digits=4))
println("  GPU shape: ", size(gpu_ssm_out))
println("  GPU norm: ", round(norm(Float32.(gpu_ssm_out)), digits=4))

# Check the actual values
println("\nFirst 10 values:")
println("  CPU: ", round.(cpu_ssm_out[1:10], digits=4))
println("  GPU: ", round.(Float32.(gpu_ssm_out[1:10]), digits=4))
