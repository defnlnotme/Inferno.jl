using Inferno
using LinearAlgebra

function debug_yg_zero()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token through layers 1-6
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
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
    @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))
    
    # Split Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    alpha_proj = ssm.ssm_alpha_weight * x_norm
    beta_proj = ssm.ssm_beta_weight * x_norm
    
    println("=== First head detailed analysis ===")
    h = 1
    g = ((h - 1) % ssm.num_k_heads) + 1
    
    qg = view(q_all, :, g)
    kg = view(k_all, :, g)
    vg = view(v_all, :, h)
    
    println("qg: ", qg)
    println("kg: ", kg)
    println("vg: ", vg)
    
    # Q/K normalization
    q_norm = sqrt(sum(abs2, qg) + ssm.ssm_norm.eps)
    k_norm = sqrt(sum(abs2, kg) + ssm.ssm_norm.eps)
    
    scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
    
    q_normalized = qg ./ q_norm .* scale
    k_normalized = kg ./ k_norm
    
    println("\nq_normalized: ", q_normalized)
    println("k_normalized: ", k_normalized)
    
    # Gates
    alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
    softplus_alpha = log(1.0 + exp(alpha_val))
    softplus_alpha = clamp(softplus_alpha, -10.0, 10.0)
    
    decay = Float32(exp(softplus_alpha * Float64(ssm.ssm_a[h])))
    decay = clamp(decay, 0.0f0, 1.0f0)
    
    beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
    beta = Float32(1.0 / (1.0 + exp(-beta_val)))
    
    println("\nalpha_val: ", alpha_val, " -> softplus: ", softplus_alpha)
    println("ssm_a[1]: ", ssm.ssm_a[1])
    println("decay = exp(softplus * ssm_a) = exp(", softplus_alpha, " * ", ssm.ssm_a[1], ") = ", decay)
    println("beta: ", beta)
    
    # State before update (should be zeros for first token)
    state = view(ssm.h, :, :, h)
    println("\nState before update:")
    println("  norm: ", sqrt(sum(abs2.(state))))
    println("  sample: ", state[:, 1])
    
    # State update
    state .*= decay
    println("State after decay:")
    println("  norm: ", sqrt(sum(abs2.(state))))
    
    # sk = k' * state
    sk = k_normalized' * state
    println("\nsk = k' * state: ", sk)
    println("  sk norm: ", sqrt(sum(abs2.(sk))))
    
    # d = beta * (v - sk)
    d = beta .* (vg .- vec(sk))
    println("\nd = beta * (v - sk):")
    println("  d: ", d)
    println("  d norm: ", sqrt(sum(abs2.(d))))
    
    # state += k * d'
    BLAS.ger!(1.0f0, k_normalized, d, state)
    println("\nState after update:")
    println("  norm: ", sqrt(sum(abs2.(state))))
    
    # Output: o = state' * q
    yg = state' * q_normalized
    println("\nyg = state' * q:")
    println("  yg norm: ", sqrt(sum(abs2.(yg))))
    println("  yg: ", yg)
end

debug_yg_zero()
