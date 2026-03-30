using Inferno
using LinearAlgebra

function debug_layer_7_spike()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    
    # Process through layers 1-6
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    println("=== Input to Layer 7 ===")
    println("x norm: ", round(sqrt(sum(abs2.(x))), digits=3))
    println("x mean: ", round(sum(x) / length(x), digits=6))
    
    layer = model.layers[7]
    ssm = layer.op
    
    # Check conv_state before layer
    println("\n=== Conv State Before Layer 7 ===")
    println("conv_state all zeros: ", all(iszero, ssm.conv_state))
    
    # Process layer 7 step by step
    x_orig = copy(x)
    x_norm = layer.in_norm(x)
    println("\nAfter in_norm: ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    
    # Projections
    qkv = ssm.in_proj * x_norm
    z = ssm.gate_proj * x_norm
    
    println("qkv norm: ", round(sqrt(sum(abs2.(qkv))), digits=3))
    println("z norm: ", round(sqrt(sum(abs2.(z))), digits=3))
    
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
    
    println("\nAfter conv + SiLU:")
    println("x_conv norm: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    
    # Split into Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    println("Q norm: ", round(sqrt(sum(abs2.(q_all))), digits=3))
    println("K norm: ", round(sqrt(sum(abs2.(k_all))), digits=3))
    println("V norm: ", round(sqrt(sum(abs2.(v_all))), digits=3))
    
    # Check alpha/beta projections
    alpha_proj = ssm.ssm_alpha_weight * x_norm
    beta_proj = ssm.ssm_beta_weight * x_norm
    
    println("\nalpha_proj norm: ", round(sqrt(sum(abs2.(alpha_proj))), digits=3))
    println("beta_proj norm: ", round(sqrt(sum(abs2.(beta_proj))), digits=3))
    
    # Check the dt_bias and A values
    println("\nssm_dt_bias sample: ", ssm.ssm_dt_bias[1:5])
    println("ssm_a sample: ", ssm.ssm_a[1:5])
    
    # Compute decay and beta for each head
    println("\nPer-head decay and beta:")
    for h in 1:min(5, ssm.num_v_heads)
        alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        decay = exp(softplus_alpha * Float64(ssm.ssm_a[h]))
        
        beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
        beta = 1.0 / (1.0 + exp(-beta_val))
        
        println("  Head $h: decay=$decay, beta=$beta")
    end
    
    # Process delta net for each head
    y_all = zeros(Float32, ssm.d_inner)
    scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
    
    println("\nPer-head output:")
    for h in 1:ssm.num_v_heads
        g = ((h - 1) % ssm.num_k_heads) + 1
        qg = view(q_all, :, g)
        kg = view(k_all, :, g)
        vg = view(v_all, :, h)
        
        q_norm = sqrt(sum(abs2, qg) + ssm.ssm_norm.eps)
        k_norm = sqrt(sum(abs2, kg) + ssm.ssm_norm.eps)
        
        q_normalized = qg ./ q_norm .* scale
        k_normalized = kg ./ k_norm
        
        alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        decay = Float32(exp(softplus_alpha * Float64(ssm.ssm_a[h])))
        
        beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
        beta = Float32(1.0 / (1.0 + exp(-beta_val)))
        
        state = view(ssm.h, :, :, h)
        state .*= decay
        
        sk = k_normalized' * state
        d = beta .* (vg .- vec(sk))
        BLAS.ger!(1.0f0, k_normalized, d, state)
        
        yg = view(y_all, (h-1)*ssm.head_v_dim+1:h*ssm.head_v_dim)
        mul!(yg, state', q_normalized)
        
        y_h_norm = sqrt(sum(abs2.(yg)))
        println("  Head $h: yg norm = $y_h_norm")
    end
    
    println("\ny_all (before norm) norm: ", round(sqrt(sum(abs2.(y_all))), digits=6))
    
    # SSM norm
    for h in 1:ssm.num_v_heads
        y_h = view(y_all, (h-1)*ssm.head_v_dim+1:h*ssm.head_v_dim)
        Inferno.ModelCPU.rmsnorm_cpu!(y_h, y_h, ssm.ssm_norm)
    end
    
    println("y_all (after norm) norm: ", round(sqrt(sum(abs2.(y_all))), digits=3))
    
    # SiLU gate
    silu_z = z .* (1.0f0 ./ (1.0f0 .+ exp.(-z)))
    y_gated = y_all .* silu_z
    
    println("y_gated norm: ", round(sqrt(sum(abs2.(y_gated))), digits=3))
    
    # Output projection
    ssm_out = ssm.ssm_out * y_gated
    println("ssm_out norm: ", round(sqrt(sum(abs2.(ssm_out))), digits=3))
    
    # Residual
    x_after_ssm = x_orig .+ ssm_out
    println("x_after_ssm norm: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
end

debug_layer_7_spike()
