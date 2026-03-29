#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

# Load model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

ssm = model.layers[1].op
println("SSM norm weight:")
println("  size: ", size(ssm.ssm_norm.weight))
println("  first 10: ", ssm.ssm_norm.weight[1:min(10, length(ssm.ssm_norm.weight))])

# Compare to expected
println("\nExpected size: head_v_dim = ", ssm.head_v_dim)
println("Actual size: ", length(ssm.ssm_norm.weight))

# Also check the GPU version
gpu_model, _ = load_model("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
gpu_ssm = gpu_model.layers[1].op
println("\nGPU SSM norm weight:")
println("  size: ", size(gpu_ssm.ssm_norm.weight))
