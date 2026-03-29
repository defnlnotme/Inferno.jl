using Inferno
using Statistics
using LinearAlgebra

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Get first SSM layer
    ssm = model.layers[1].op
    in_norm = model.layers[1].in_norm

    # Input: token 760 "The"
    x = copy(view(model.embed, :, 760))

    # Normalize
    h = in_norm(x)
    
    # Input projections
    qkv = ssm.in_proj * h
    z = ssm.gate_proj * h

    # Conv state update
    ssm.conv_state[:, 1:3] .= ssm.conv_state[:, 2:4]
    ssm.conv_state[:, 4] .= qkv

    # Convolution
    x_conv = zeros(Float32, ssm.conv_channels)
    for k in 1:ssm.conv_kernel
        for c in 1:ssm.conv_channels
            x_conv[c] += ssm.conv_state[c, k] * ssm.ssm_conv1d[k, c]
        end
    end

    # SiLU
    @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))

    # Split Q/K/V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(x_conv[1:qk_size], ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(x_conv[qk_size+1:2*qk_size], ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(x_conv[2*qk_size+1:2*qk_size+ssm.d_inner], ssm.head_v_dim, ssm.num_v_heads)

    # Alpha/beta
    alpha_proj = ssm.ssm_alpha_weight * h
    beta_proj = ssm.ssm_beta_weight * h

    println("alpha_proj: ", alpha_proj)
    println("beta_proj: ", beta_proj)
    println("ssm_dt_bias: ", ssm.ssm_dt_bias)
    println("ssm_a: ", ssm.ssm_a)

    # Process first head
    h_idx = 1
    g = ((h_idx - 1) % ssm.num_k_heads) + 1
    
    qg = view(q_all, :, g)
    kg = view(k_all, :, g)
    vg = view(v_all, :, h_idx)

    println("\nHead 1:")
    println("  qg norm: ", sqrt(sum(abs2, qg)))
    println("  kg norm: ", sqrt(sum(abs2, kg)))
    println("  vg norm: ", sqrt(sum(abs2, vg)))

    # Q/K normalization
    q_norm = sqrt(sum(abs2, qg) + 1e-6)
    k_norm = sqrt(sum(abs2, kg) + 1e-6)
    scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
    
    q_normalized = qg ./ q_norm .* scale
    k_normalized = kg ./ k_norm

    println("  q_normalized[1:3]: ", q_normalized[1:3])
    println("  k_normalized[1:3]: ", k_normalized[1:3])

    # Alpha/beta computation
    alpha_val = Float64(alpha_proj[h_idx]) + Float64(ssm.ssm_dt_bias[h_idx])
    alpha_val = clamp(alpha_val, -20.0, 20.0)
    softplus_alpha = log(1.0 + exp(alpha_val))
    softplus_alpha = clamp(softplus_alpha, -10.0, 10.0)
    decay = Float32(exp(softplus_alpha * Float64(ssm.ssm_a[h_idx])))
    decay = clamp(decay, 0.0f0, 1.0f0)

    beta_val = Float64(beta_proj[h_idx])
    beta_val = clamp(beta_val, -20.0, 20.0)
    beta = Float32(1.0 / (1.0 + exp(-beta_val)))

    println("\n  alpha_val: ", alpha_val)
    println("  softplus_alpha: ", softplus_alpha)
    println("  decay: ", decay)
    println("  beta: ", beta)
end

main()
