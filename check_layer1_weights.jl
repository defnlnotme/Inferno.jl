using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check layer 1 (first SSM layer)
layer1 = model.layers[1]
ssm = layer1.op

println("=== Layer 1 SSM Weights ===")
println("in_proj shape: ", size(ssm.in_proj))
println("in_proj norm: ", round(norm(ssm.in_proj), digits=5))

println("\ngate_proj shape: ", size(ssm.gate_proj))
println("gate_proj norm: ", round(norm(ssm.gate_proj), digits=5))

println("\nssm_a shape: ", size(ssm.ssm_a))
println("ssm_a values: ", round.(ssm.ssm_a, digits=5))

println("\nssm_conv1d shape: ", size(ssm.ssm_conv1d))
println("ssm_conv1d norm: ", round(norm(ssm.ssm_conv1d), digits=5))

# Check output projection
println("\nssm_out shape: ", size(ssm.ssm_out))
println("ssm_out norm: ", round(norm(ssm.ssm_out), digits=5))

# Check if there's a shape mismatch
println("\n=== Shape Check ===")
println("Expected in_proj: (2048, 1024) for hidden=1024, d_inner=2048")
println("Actual in_proj: ", size(ssm.in_proj))
