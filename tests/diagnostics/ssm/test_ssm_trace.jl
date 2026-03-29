#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get embedding for "The"
x = model.embed[:, 562]  # Token 562 = "ĠThe"
println("Input x: norm=$(round(norm(x), digits=3)), first 5=$(round.(x[1:5], digits=4))")

# Apply input norm
layer = model.layers[1]
x_normed = layer.in_norm(x)
println("\nAfter in_norm: norm=$(round(norm(x_normed), digits=3)), first 5=$(round.(x_normed[1:5], digits=4))")

# SSM computation step by step
ssm = layer.op

# 1. Input projection
qkv = ssm.in_proj * x_normed
println("\nqkv: size=$(size(qkv)), norm=$(round(norm(qkv), digits=3)), first 5=$(round.(qkv[1:5], digits=4))")

# 2. Gate projection
z = ssm.gate_proj * x_normed
println("z: size=$(size(z)), norm=$(round(norm(z), digits=3)), first 5=$(round.(z[1:5], digits=4))")

# 3. Conv state update (position 0)
# At pos=0, conv_state is all zeros initially
println("\nconv_state before: $(sum(abs.(ssm.conv_state)))")

# 4. Compute convolution
x_conv = zeros(Float32, ssm.conv_channels)
for k in 1:ssm.conv_kernel
    for c in 1:ssm.conv_channels
        x_conv[c] += ssm.conv_state[c, k] * ssm.ssm_conv1d[k, c]
    end
end
println("x_conv after conv: norm=$(round(norm(x_conv), digits=3)), first 5=$(round.(x_conv[1:5], digits=4))")

# After first token, conv_state is updated
# conv_state[:, conv_kernel] = qkv at pos 0
# For next position, we shift: conv_state[:, 1:(kernel-1)] = conv_state[:, 2:kernel]

# Let's check the split dimensions
qk_size = ssm.head_k_dim * ssm.num_k_heads
println("\nqk_size = head_k_dim * num_k_heads = $(ssm.head_k_dim) * $(ssm.num_k_heads) = $qk_size")
println("d_inner = $(ssm.d_inner)")
println("Expected conv_channels = d_inner + 2*qk_size = $(ssm.d_inner) + 2*$qk_size = $(ssm.d_inner + 2*qk_size)")
println("Actual conv_channels = $(ssm.conv_channels)")

# Check the state dimension
println("\nState h: size=$(size(ssm.h))")
println("Expected: (head_v_dim, head_k_dim, num_v_heads) = ($(ssm.head_v_dim), $(ssm.head_k_dim), $(ssm.num_v_heads))")
