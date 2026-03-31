using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check layer 1 SSM
layer1 = model.layers[1]
ssm = layer1.op

println("=== SSM Configuration ===")
println("d_inner: ", ssm.d_inner)
println("conv_kernel: ", ssm.conv_kernel)
println("conv_channels: ", ssm.conv_channels)
println("num_v_heads: ", ssm.num_v_heads)
println("num_k_heads: ", ssm.num_k_heads)
println("head_k_dim: ", ssm.head_k_dim)
println("head_v_dim: ", ssm.head_v_dim)

println("\n=== Weight Shapes ===")
println("in_proj: ", size(ssm.in_proj), " (expected: (d_inner*3, hidden) = (6144, 1024))")
println("gate_proj: ", size(ssm.gate_proj), " (expected: (d_inner, hidden) = (2048, 1024))")
println("ssm_conv1d: ", size(ssm.ssm_conv1d), " (expected: (kernel, channels) = (4, 6144))")
println("ssm_out: ", size(ssm.ssm_out), " (expected: (hidden, d_inner) = (1024, 2048))")

println("\n=== Conv1D Check ===")
println("ssm_conv1d first column (kernel for channel 1): ", round.(ssm.ssm_conv1d[:, 1], digits=5))

# Check if conv1d looks like a causal kernel (should have larger values at the end)
println("\nConv1d norm per position:")
for k in 1:ssm.conv_kernel
    println("  position $k: ", round(norm(ssm.ssm_conv1d[k, :]), digits=5))
end

# Check alpha/beta
println("\n=== Alpha/Beta Check ===")
println("ssm_alpha_weight shape: ", size(ssm.ssm_alpha_weight))
println("ssm_beta_weight shape: ", size(ssm.ssm_beta_weight))
println("ssm_a: ", round.(ssm.ssm_a, digits=5))
println("ssm_dt_bias: ", round.(ssm.ssm_dt_bias, digits=5))
