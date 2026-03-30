using Inferno
using LinearAlgebra

function investigate_output_projection()
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
    
    # Get layer 7
    layer = model.layers[7]
    ssm = layer.op
    
    # Process SSM step by step
    x_norm = layer.in_norm(x)
    
    # Full SSM computation
    qkv = ssm.in_proj * x_norm
    z = ssm.gate_proj * x_norm
    
    # Conv
    if ssm.conv_kernel > 1
        ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
    end
    ssm.conv_state[:, ssm.conv_kernel] .= qkv
    
    x_conv = Vector{Float32}(undef, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
    end
    x_conv .= x_conv .* (1.0f0 ./ (1.0f0 .+ exp.(-x_conv)))
    
    # Split
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    alpha_proj = ssm.ssm_alpha_weight * x_norm
    beta_proj = ssm.ssm_beta_weight * x_norm
    
    # Process all heads
    y_all = zeros(Float32, ssm.d_inner)
    scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
    
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
    end
    
    println("=== Output Projection Investigation ===")
    
    println("\n--- Before SSM Norm ---")
    println("y_all norm: ", sqrt(sum(abs2.(y_all))))
    println("y_all mean: ", sum(y_all) / length(y_all))
    println("y_all sample: ", y_all[1:5])
    
    # Check the variance
    mean_y = sum(y_all) / length(y_all)
    var_y = sum((y_all .- mean_y).^2) / length(y_all)
    println("y_all variance: ", var_y)
    
    println("\n--- SSM Norm ---")
    println("ssm_norm.weight shape: ", size(ssm.ssm_norm.weight))
    println("ssm_norm.weight mean: ", sum(ssm.ssm_norm.weight) / length(ssm.ssm_norm.weight))
    println("ssm_norm.eps: ", ssm.ssm_norm.eps)
    
    # Apply SSM norm per-head
    for h in 1:ssm.num_v_heads
        y_h = view(y_all, (h-1)*ssm.head_v_dim+1:h*ssm.head_v_dim)
        Inferno.ModelCPU.rmsnorm_cpu!(y_h, y_h, ssm.ssm_norm)
    end
    
    println("\n--- After SSM Norm ---")
    println("y_all norm: ", sqrt(sum(abs2.(y_all))))
    println("y_all mean: ", sum(y_all) / length(y_all))
    println("y_all sample: ", y_all[1:5])
    
    println("\n--- SiLU Gate ---")
    println("z norm: ", sqrt(sum(abs2.(z))))
    silu_z = z .* (1.0f0 ./ (1.0f0 .+ exp.(-z)))
    println("silu(z) norm: ", sqrt(sum(abs2.(silu_z))))
    
    y_gated = y_all .* silu_z
    
    println("\n--- After SiLU Gate ---")
    println("y_gated norm: ", sqrt(sum(abs2.(y_gated))))
    
    println("\n--- Output Projection ---")
    println("ssm_out weight shape: ", size(ssm.ssm_out))
    println("ssm_out weight norm: ", sqrt(sum(abs2.(ssm.ssm_out))))
    
    ssm_output = ssm.ssm_out * y_gated
    
    println("\n--- SSM Output ---")
    println("ssm_output norm: ", sqrt(sum(abs2.(ssm_output))))
    println("ssm_output sample: ", ssm_output[1:5])
    
    println("\n=== Key Observation ===")
    println("The SSM norm amplifies y_all by a large factor.")
    println("This is because y_all has very small values (norm ~0.02)")
    println("and RMS norm divides by sqrt(mean(y^2) + eps).")
    println("")
    println("For y_all with norm ~0.02 across 2048 elements:")
    println("  mean(y^2) = 0.02^2 / 2048 = 1.95e-7")
    println("  scale = 1 / sqrt(1.95e-7 + 1e-6) = 912")
    println("  y_normed norm = 0.02 * 912 * weight_mean = 18.2")
    println("")
    println("This is expected behavior - small inputs get amplified.")
end

investigate_output_projection()
