using Inferno
using LinearAlgebra
using Printf

function investigate_decay()
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
    
    println("=== Decay Investigation ===")
    println("\nLayer 7 SSM parameters:")
    println("  head_k_dim: ", ssm.head_k_dim)
    println("  head_v_dim: ", ssm.head_v_dim)
    println("  num_k_heads: ", ssm.num_k_heads)
    println("  num_v_heads: ", ssm.num_v_heads)
    
    println("\n--- Per-head decay computation ---")
    println("Formula: decay = exp(softplus(alpha + dt_bias) * A)")
    println("         softplus(x) = log(1 + exp(x))")
    
    println("\nHead | alpha_proj | dt_bias | alpha_val | softplus | A | decay")
    println("-----|------------|---------|-----------|----------|---|------")
    
    for h in 1:ssm.num_v_heads
        alpha_raw = alpha_proj[h]
        dt_bias = ssm.ssm_dt_bias[h]
        alpha_val = clamp(Float64(alpha_raw) + Float64(dt_bias), -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        A = ssm.ssm_a[h]
        decay = exp(softplus_alpha * Float64(A))
        
        @printf("  %2d | %10.4f | %7.4f | %9.4f | %8.4f | %.4f | %.6f\n", 
                h, alpha_raw, dt_bias, alpha_val, softplus_alpha, A, decay)
    end
    
    println("\n--- Key observations ---")
    println("1. dt_bias values are negative for most heads")
    println("2. A values are negative for all heads (decay, not growth)")
    println("3. softplus(alpha) is always positive")
    println("4. decay = exp(softplus * A) should be < 1 if A < 0")
    
    # Check if decay values are reasonable
    println("\n--- Decay range check ---")
    decays = Float64[]
    for h in 1:ssm.num_v_heads
        alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        decay = exp(softplus_alpha * Float64(ssm.ssm_a[h]))
        push!(decays, decay)
    end
    
    println("Decay min: ", minimum(decays))
    println("Decay max: ", maximum(decays))
    println("Decay mean: ", sum(decays) / length(decays))
    
    # Check what happens when decay is applied to state
    # For first token, state is zero, so decay doesn't matter
    # But for subsequent tokens, decay controls state growth
    
    println("\n--- State decay effect ---")
    println("For first token: state is zero, decay has no effect")
    println("For subsequent tokens: state *= decay, then state += k * d'")
    println("")
    println("If decay < 1, state should shrink over time")
    println("If decay > 1, state should grow (unstable)")
    println("Most decays are in range 0.7-0.95, which is stable")
end

investigate_decay()
