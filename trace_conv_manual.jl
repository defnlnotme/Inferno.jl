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
println("x_norm norm: ", round(norm(x_norm), digits=5))

# Input projections
qkv = ssm.in_proj * x_norm
z = ssm.gate_proj * x_norm

println("\n=== Manual Convolution ===")
println("qkv norm: ", round(norm(qkv), digits=5))
println("ssm_conv1d shape: ", size(ssm.ssm_conv1d))
println("ssm_conv1d norm: ", round(norm(ssm.ssm_conv1d), digits=5))

# Manual conv state update (copying the code logic)
if ssm.conv_kernel > 1
    ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
end
ssm.conv_state[:, ssm.conv_kernel] .= qkv

println("\nAfter update, conv_state norm: ", round(norm(ssm.conv_state), digits=5))

# Compute convolution
x_conv = Vector{Float32}(undef, ssm.conv_channels)
for c in 1:ssm.conv_channels
    x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
end

println("\nx_conv norm: ", round(norm(x_conv), digits=5))
println("x_conv[1:5]: ", round.(x_conv[1:5], digits=5))

# After SiLU
x_conv_silu = x_conv .* (1.0f0 ./ (1.0f0 .+ exp.(-x_conv)))
println("\nx_conv_silu norm: ", round(norm(x_conv_silu), digits=5))
println("x_conv_silu[1:5]: ", round.(x_conv_silu[1:5], digits=5))
