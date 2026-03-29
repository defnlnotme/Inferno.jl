module DebugSSM
using Inferno

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("=== Layer 0 SSM weight shapes ===")

# Load weights
prefix = "blk.0"

in_proj = Float32.(LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_qkv.weight"))'
gate_proj = Float32.(LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_gate.weight"))'
ssm_out = Float32.(LoaderCPU.extract_tensor_cpu(file, "$(prefix).ssm_out.weight"))'
ssm_conv1d = Float32.(LoaderCPU.extract_tensor_cpu(file, "$(prefix).ssm_conv1d.weight"))'
ssm_alpha_weight = Float32.(LoaderCPU.extract_tensor_cpu(file, "$(prefix).ssm_alpha.weight"))'
ssm_beta_weight = Float32.(LoaderCPU.extract_tensor_cpu(file, "$(prefix).ssm_beta.weight"))'
ssm_a = Float32.(vec(LoaderCPU.extract_tensor_cpu(file, "$(prefix).ssm_a")))
ssm_dt_bias = Float32.(vec(LoaderCPU.extract_tensor_cpu(file, "$(prefix).ssm_dt.bias")))

println("in_proj (attn_qkv): ", size(in_proj))
println("gate_proj: ", size(gate_proj))
println("ssm_out: ", size(ssm_out))
println("ssm_conv1d: ", size(ssm_conv1d))
println("ssm_alpha_weight: ", size(ssm_alpha_weight))
println("ssm_beta_weight: ", size(ssm_beta_weight))
println("ssm_a: ", size(ssm_a))
println("ssm_dt_bias: ", size(ssm_dt_bias))

println("\n=== Derived dimensions ===")
num_v_heads = length(ssm_a)
num_k_heads = num_v_heads
d_inner = size(gate_proj, 1)
head_v_dim = d_inner ÷ num_v_heads
conv_channels = size(in_proj, 1)
conv_kernel = size(ssm_conv1d, 2)
head_k_dim = (conv_channels - d_inner) ÷ (2 * num_k_heads)

println("num_v_heads: ", num_v_heads)
println("num_k_heads: ", num_k_heads)
println("d_inner: ", d_inner)
println("head_v_dim: ", head_v_dim)
println("conv_channels: ", conv_channels)
println("conv_kernel: ", conv_kernel)
println("head_k_dim: ", head_k_dim)

println("\n=== Check ssm_a values ===")
println("ssm_a: ", ssm_a)

println("\n=== Conv1d weight shape ===")
println("Expected: (conv_channels=$conv_channels, kernel=$conv_kernel)")
println("Actual: ", size(ssm_conv1d))
end
