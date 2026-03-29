using Inferno

# Load model on both CPU and GPU
cpu_model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
gpu_model, _ = load_model("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("Comparing embedding weights:")
println("CPU embed[1:5, 1]: ", cpu_model.embed[1:5, 1])
println("GPU embed[1:5, 1]: ", Float32.(gpu_model.embed[1:5, 1]))

println("\nComparing first SSM layer weights:")
cpu_ssm = cpu_model.layers[1].op
gpu_ssm = gpu_model.layers[1].op

# Compare in_proj
println("CPU in_proj[1:5, 1]: ", cpu_ssm.in_proj[1:5, 1])
println("GPU in_proj[1:5, 1]: ", Float32.(gpu_ssm.in_proj[1:5, 1]))

# Compare gate_proj
println("\nCPU gate_proj[1:5, 1]: ", cpu_ssm.gate_proj[1:5, 1])
println("GPU gate_proj[1:5, 1]: ", Float32.(gpu_ssm.gate_proj[1:5, 1]))

# Compare conv1d
println("\nCPU ssm_conv1d size: ", size(cpu_ssm.ssm_conv1d))
println("CPU ssm_conv1d[:, 1]: ", cpu_ssm.ssm_conv1d[:, 1])
println("GPU ssm_conv1d_weight_cpu shape: ", size(gpu_ssm.ssm_conv1d_weight_cpu))
println("GPU ssm_conv1d_weight_cpu[:, 1]: ", Float32.(gpu_ssm.ssm_conv1d_weight_cpu[:, 1]))

# Compare MLP weights
println("\n=== MLP weights ===")
cpu_mlp = cpu_model.layers[1].mlp
gpu_mlp = gpu_model.layers[1].mlp

println("CPU mlp.gate_weight[1:5, 1]: ", cpu_mlp.gate_weight[1:5, 1])
println("GPU mlp.gate_weight[1:5, 1]: ", Float32.(gpu_mlp.gate_weight[1:5, 1]))

println("\nCPU mlp.up_weight[1:5, 1]: ", cpu_mlp.up_weight[1:5, 1])
println("GPU mlp.up_weight[1:5, 1]: ", Float32.(gpu_mlp.up_weight[1:5, 1]))

println("\nCPU mlp.down_weight[1, 1:5]: ", cpu_mlp.down_weight[1, 1:5])
println("GPU mlp.down_weight[1, 1:5]: ", Float32.(gpu_mlp.down_weight[1, 1:5]))
