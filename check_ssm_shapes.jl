using Inferno
using LinearAlgebra

# Load model and check SSM weight shapes in detail
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check layer 0 (SSM)
layer0 = model.layers[1]
ssm = layer0.op

println("=== SSM Layer 0 Weight Shapes ===")
println("in_proj: ", size(ssm.in_proj))
println("  Expected: (conv_channels, hidden) = (6144, 1024)")
println("  Comment says: (hidden, conv_channels)")
println()

println("gate_proj: ", size(ssm.gate_proj))
println("  Expected: (d_inner, hidden) = (2048, 1024)")
println("  Comment says: (hidden, d_inner)")
println()

println("ssm_out: ", size(ssm.ssm_out))
println("  Expected: (hidden, d_inner) = (1024, 2048)")
println("  Comment says: (d_inner, hidden)")
println()

println("ssm_conv1d: ", size(ssm.ssm_conv1d))
println("  Expected: (conv_kernel, conv_channels) = (4, 6144)")
println()

println("ssm_alpha_weight: ", size(ssm.ssm_alpha_weight))
println("  Expected: (num_v_heads, hidden) = (16, 1024)")
println("  Comment says: (hidden, num_v_heads)")
println()

println("ssm_beta_weight: ", size(ssm.ssm_beta_weight))
println()

println("\n=== Dimensions ===")
println("num_v_heads: ", ssm.num_v_heads)
println("num_k_heads: ", ssm.num_k_heads)
println("head_k_dim: ", ssm.head_k_dim)
println("head_v_dim: ", ssm.head_v_dim)
println("d_inner: ", ssm.d_inner)
println("conv_channels: ", ssm.conv_channels)
println("conv_kernel: ", ssm.conv_kernel)

# Test matrix multiplication dimensions
println("\n=== Matrix Mult Test ===")
x = randn(Float32, 1024)
println("x shape: ", size(x))
println("in_proj * x: ", size(ssm.in_proj * x))
println("gate_proj * x: ", size(ssm.gate_proj * x))
