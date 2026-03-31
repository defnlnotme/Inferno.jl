using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get "The" embedding
token = 761
x = model.embed[:, token + 1]

# Trace through layer 1 (SSM)
layer1 = model.layers[1]
ssm = layer1.op

# Input norm
x_norm = layer1.in_norm(x)
println("=== Input ===")
println("x norm: ", round(norm(x), digits=5))
println("x_norm norm: ", round(norm(x_norm), digits=5))

# Input projections
qkv = ssm.in_proj * x_norm
z = ssm.gate_proj * x_norm

println("\n=== Projections ===")
println("qkv norm: ", round(norm(qkv), digits=5))
println("z norm: ", round(norm(z), digits=5))

# Convolution
# Initially conv_state is zeros, so x_conv will be small
x_conv = similar(qkv)
for c in 1:ssm.conv_channels
    x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
end

println("\n=== Convolution ===")
println("conv_state norm: ", round(norm(ssm.conv_state), digits=5))
println("x_conv norm: ", round(norm(x_conv), digits=5))

# After SiLU
x_conv_silu = x_conv .* (1.0f0 ./ (1.0f0 .+ exp.(-x_conv)))
println("x_conv_silu norm: ", round(norm(x_conv_silu), digits=5))

# Q/K/V split
qk_size = ssm.head_k_dim * ssm.num_k_heads
q_all = reshape(view(x_conv_silu, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
k_all = reshape(view(x_conv_silu, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
v_all = reshape(view(x_conv_silu, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)

println("\n=== Q/K/V ===")
println("q_all norm: ", round(norm(q_all), digits=5))
println("k_all norm: ", round(norm(k_all), digits=5))
println("v_all norm: ", round(norm(v_all), digits=5))

# Alpha/beta
alpha_proj = ssm.ssm_alpha_weight * x_norm
beta_proj = ssm.ssm_beta_weight * x_norm

println("\n=== Alpha/Beta ===")
println("alpha_proj norm: ", round(norm(alpha_proj), digits=5))
println("beta_proj norm: ", round(norm(beta_proj), digits=5))

# Check alpha/beta values
println("\nalpha values: ", round.(alpha_proj[1:5], digits=5))
println("beta values: ", round.(beta_proj[1:5], digits=5))

# Check ssm_a (decay parameter)
println("\nssm_a values: ", round.(ssm.ssm_a[1:5], digits=5))
println("ssm_dt_bias values: ", round.(ssm.ssm_dt_bias[1:5], digits=5))
