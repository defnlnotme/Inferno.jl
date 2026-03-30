using Inferno
using LinearAlgebra

function investigate_state_accumulation()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    
    # Process through first 6 layers
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    # Focus on layer 7
    layer = model.layers[7]
    ssm = layer.op
    x_norm = layer.in_norm(x)
    
    # Get projections
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
    
    # Split Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    # Alpha/beta projections
    alpha_proj = ssm.ssm_alpha_weight * x_norm
    beta_proj = ssm.ssm_beta_weight * x_norm
    
    println("=== State Accumulation Investigation ===")
    
    # Focus on head 1
    h = 1
    g = 1
    
    qg = view(q_all, :, g)
    kg = view(k_all, :, g)
    vg = view(v_all, :, h)
    
    q_norm = sqrt(sum(abs2, qg) + ssm.ssm_norm.eps)
    k_norm = sqrt(sum(abs2, kg) + ssm.ssm_norm.eps)
    
    scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
    q_normalized = qg ./ q_norm .* scale
    k_normalized = kg ./ k_norm
    
    alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
    softplus_alpha = log(1.0 + exp(alpha_val))
    decay = Float32(exp(softplus_alpha * Float64(ssm.ssm_a[h])))
    
    beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
    beta = Float32(1.0 / (1.0 + exp(-beta_val)))
    
    println("\n--- Head 1 State Update ---")
    println("Before update:")
    println("  State norm: ", sqrt(sum(abs2.(ssm.h[:, :, h]))))
    println("  Decay: ", decay)
    println("  Beta: ", beta)
    
    # Apply decay
    state = view(ssm.h, :, :, h)
    state .*= decay
    
    println("\nAfter decay:")
    println("  State norm: ", sqrt(sum(abs2.(state))))
    
    # Compute sk = k' * state
    sk = k_normalized' * state
    
    println("\nsk = k' * state:")
    println("  sk norm: ", sqrt(sum(abs2.(sk))))
    println("  sk sample: ", sk[1:3])
    
    # Compute d = beta * (v - sk)
    d = beta .* (vg .- vec(sk))
    
    println("\nd = beta * (v - sk):")
    println("  v norm: ", sqrt(sum(abs2.(vg))))
    println("  d norm: ", sqrt(sum(abs2.(d))))
    println("  d sample: ", d[1:3])
    
    # Update state: state += k * d'
    println("\nBefore state update (outer product):")
    println("  k_normalized norm: ", sqrt(sum(abs2.(k_normalized))))
    println("  d norm: ", sqrt(sum(abs2.(d))))
    
    # The outer product k * d' has norm = norm(k) * norm(d)
    outer_norm = sqrt(sum(abs2.(k_normalized))) * sqrt(sum(abs2.(d)))
    println("  Expected outer product norm: ", outer_norm)
    
    BLAS.ger!(1.0f0, k_normalized, d, state)
    
    println("\nAfter state update:")
    println("  State norm: ", sqrt(sum(abs2.(state))))
    
    # Compute output: yg = state' * q
    yg = state' * q_normalized
    
    println("\nOutput yg = state' * q:")
    println("  yg norm: ", sqrt(sum(abs2.(yg))))
    println("  yg sample: ", yg[1:3])
    
    println("\n=== Analysis ===")
    println("1. For first token, state is zero before update")
    println("2. After decay, state is still zero (decay of zero is zero)")
    println("3. sk = k' * state = k' * 0 = 0")
    println("4. d = beta * (v - 0) = beta * v")
    println("5. state += k * d' = k * (beta * v)' = beta * k * v'")
    println("6. yg = state' * q = (beta * k * v')' * q = beta * v * k' * q")
    println("")
    println("This means for first token:")
    println("  yg = beta * v * (k' * q)")
    println("     = beta * v * dot(k, q)")
    println("")
    println("Since k and q are both derived from x_conv,")
    println("and x_conv is small (norm ~4.5), the output yg is small.")
end

investigate_state_accumulation()
