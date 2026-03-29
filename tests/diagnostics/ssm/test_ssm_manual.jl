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

# Check initial state
println("Initial SSM state:")
println("  conv_state: size=$(size(ssm.conv_state)), all_zeros=$(all(iszero, ssm.conv_state))")
println("  h: size=$(size(ssm.h)), all_zeros=$(all(iszero, ssm.h))")
println("  h initial sample values: ", ssm.h[1:3, 1:3, 1])

# Get embedding
x = model.embed[:, 562]  # " The"
println("\nInput embedding:")
println("  norm: ", round(norm(x), digits=4))

# Apply input norm
x_normed = model.layers[1].in_norm(x)
println("\nAfter input_norm:")
println("  norm: ", round(norm(x_normed), digits=4))

# Create cache
config = model.config
cache = Inferno.ModelCPU.init_kv_cache_cpu(config, 512)

# Step through SSM manually
println("\n=== Manual SSM trace ===")

# 1. Input projections
qkv = ssm.in_proj * x_normed
z = ssm.gate_proj * x_normed
println("1. Input projections:")
println("   qkv: size=$(size(qkv)), norm=$(round(norm(qkv), digits=4))")
println("   z: size=$(size(z)), norm=$(round(norm(z), digits=4))")

# 2. Update conv state (ring buffer)
if ssm.conv_kernel > 1
    ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
end
ssm.conv_state[:, ssm.conv_kernel] .= qkv

println("\n2. Conv state after update:")
println("   conv_state[:, end] sample: ", round.(ssm.conv_state[1:5, end], digits=4))

# 3. Compute convolution
x_conv = zeros(Float32, ssm.conv_channels)
for k in 1:ssm.conv_kernel
    for c in 1:ssm.conv_channels
        x_conv[c] += ssm.conv_state[c, k] * ssm.ssm_conv1d[k, c]
    end
end
println("\n3. After conv1d:")
println("   x_conv: size=$(size(x_conv)), norm=$(round(norm(x_conv), digits=4))")
println("   x_conv sample: ", round.(x_conv[1:5], digits=4))

# 4. SiLU activation
@. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))
println("\n4. After SiLU:")
println("   x_conv norm: ", round(norm(x_conv), digits=4))
println("   x_conv sample: ", round.(x_conv[1:5], digits=4))

# 5. Split into Q, K, V
qk_size = ssm.head_k_dim * ssm.num_k_heads
q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)

println("\n5. Q/K/V split:")
println("   q_all: size=$(size(q_all)), norm=$(round(norm(q_all), digits=4))")
println("   k_all: size=$(size(k_all)), norm=$(round(norm(k_all), digits=4))")
println("   v_all: size=$(size(v_all)), norm=$(round(norm(v_all), digits=4))")

# 6. Alpha/beta projections
alpha_proj = ssm.ssm_alpha_weight * x_normed
beta_proj = ssm.ssm_beta_weight * x_normed
println("\n6. Alpha/beta:")
println("   alpha_proj: size=$(size(alpha_proj)), sample=$(round.(alpha_proj[1:3], digits=4))")
println("   beta_proj: size=$(size(beta_proj)), sample=$(round.(beta_proj[1:3], digits=4))")

# 7. Process first head
println("\n7. Processing first head:")
h = 1
g = ((h - 1) % ssm.num_k_heads) + 1

qg = view(q_all, :, g)
kg = view(k_all, :, g)
vg = view(v_all, :, h)

println("   qg norm: ", round(norm(qg), digits=4))
println("   kg norm: ", round(norm(kg), digits=4))
println("   vg norm: ", round(norm(vg), digits=4))

# Q/K L2 normalization
q_norm = sqrt(sum(abs2, qg) + Float32(1e-6))
k_norm = sqrt(sum(abs2, kg) + Float32(1e-6))

q_normalized = qg ./ q_norm ./ sqrt(Float32(ssm.head_k_dim))
k_normalized = kg ./ k_norm

println("   q_normalized norm: ", round(norm(q_normalized), digits=4))
println("   k_normalized norm: ", round(norm(k_normalized), digits=4))

# Gate values
alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
softplus_alpha = log(1.0 + exp(alpha_val))
softplus_alpha = clamp(softplus_alpha, -10.0, 10.0)

decay = Float32(exp(softplus_alpha * Float64(ssm.ssm_a[h])))
decay = clamp(decay, 0.0f0, 1.0f0)

beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
beta = Float32(1.0 / (1.0 + exp(-beta_val)))

println("   alpha_val: ", round(alpha_val, digits=4))
println("   softplus_alpha: ", round(softplus_alpha, digits=4))
println("   decay: ", round(decay, digits=4))
println("   beta: ", round(beta, digits=4))
