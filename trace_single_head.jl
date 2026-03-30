using Inferno
using LinearAlgebra

function trace_single_head()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # First token
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    layer = model.layers[7]
    ssm = layer.op
    x_norm = layer.in_norm(x)
    
    # Get qkv
    qkv = ssm.in_proj * x_norm
    z = ssm.gate_proj * x_norm
    
    # Convolution
    if ssm.conv_kernel > 1
        ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
    end
    ssm.conv_state[:, ssm.conv_kernel] .= qkv
    
    x_conv = Vector{Float32}(undef, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
    end
    
    # SiLU
    x_conv .= x_conv .* (1.0f0 ./ (1.0f0 .+ exp.(-x_conv)))
    
    # Split
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    # Alpha/beta
    alpha_proj = ssm.ssm_alpha_weight * x_norm
    beta_proj = ssm.ssm_beta_weight * x_norm
    
    # Focus on head 1
    h = 1
    g = 1  # (h-1) % num_k_heads + 1 = 1
    qg = view(q_all, :, g)
    kg = view(k_all, :, g)
    vg = view(v_all, :, h)
    
    println("=== Head 1 Detailed Trace ===")
    println("\nqg (before norm): ", round.(qg[1:5], digits=4))
    println("kg (before norm): ", round.(kg[1:5], digits=4))
    println("vg: ", round.(vg[1:5], digits=4))
    
    q_norm = sqrt(sum(abs2, qg) + ssm.ssm_norm.eps)
    k_norm = sqrt(sum(abs2, kg) + ssm.ssm_norm.eps)
    
    println("\nq_norm: ", q_norm)
    println("k_norm: ", k_norm)
    
    scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
    q_normalized = qg ./ q_norm .* scale
    k_normalized = kg ./ k_norm
    
    println("\nq_normalized norm: ", sqrt(sum(abs2.(q_normalized))))
    println("k_normalized norm: ", sqrt(sum(abs2.(k_normalized))))
    println("q_normalized sample: ", round.(q_normalized[1:5], digits=6))
    println("k_normalized sample: ", round.(k_normalized[1:5], digits=6))
    
    alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
    softplus_alpha = log(1.0 + exp(alpha_val))
    decay = Float32(exp(softplus_alpha * Float64(ssm.ssm_a[h])))
    
    beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
    beta = Float32(1.0 / (1.0 + exp(-beta_val)))
    
    println("\nalpha_proj[$h]: ", alpha_proj[h])
    println("ssm_dt_bias[$h]: ", ssm.ssm_dt_bias[h])
    println("alpha_val: ", alpha_val)
    println("softplus_alpha: ", softplus_alpha)
    println("ssm_a[$h]: ", ssm.ssm_a[h])
    println("decay: ", decay)
    println("\nbeta_proj[$h]: ", beta_proj[h])
    println("beta: ", beta)
    
    state = view(ssm.h, :, :, h)
    
    println("\nState before decay:")
    println("  norm: ", sqrt(sum(abs2.(state))))
    println("  sample [1:3, 1:3]:")
    println("  ", round.(state[1:3, 1:3], digits=6))
    
    state .*= decay
    
    println("\nState after decay:")
    println("  norm: ", sqrt(sum(abs2.(state))))
    
    # For first token, state should be zeros
    println("\nState is all zeros? ", all(iszero, state))
    
    # Compute sk
    sk = k_normalized' * state
    println("\nsk = k' * state:")
    println("  sk norm: ", sqrt(sum(abs2.(sk))))
    println("  sk sample: ", round.(sk[1:5], digits=8))
    
    # Compute d
    d = beta .* (vg .- vec(sk))
    println("\nd = beta * (v - sk):")
    println("  d norm: ", sqrt(sum(abs2.(d))))
    println("  d sample: ", round.(d[1:5], digits=6))
    
    # Update state
    BLAS.ger!(1.0f0, k_normalized, d, state)
    
    println("\nState after update:")
    println("  norm: ", sqrt(sum(abs2.(state))))
    println("  sample [1:3, 1:3]:")
    println("  ", round.(state[1:3, 1:3], digits=6))
    
    # Compute output
    yg = state' * q_normalized
    
    println("\nyg = state' * q:")
    println("  yg norm: ", sqrt(sum(abs2.(yg))))
    println("  yg sample: ", round.(yg[1:5], digits=8))
end

trace_single_head()
