using Inferno
using LinearAlgebra

function trace_gate_silu_output()
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
    
    println("=== Layer 7 Detailed Trace ===")
    println("Input norm: ", round(sqrt(sum(abs2.(x))), digits=3))
    
    layer = model.layers[7]
    ssm = layer.op
    x_orig = copy(x)
    
    # Step 1: Input norm
    x_norm = layer.in_norm(x)
    println("\n1. After in_norm: ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    
    # Step 2: Projections
    qkv = ssm.in_proj * x_norm
    z = ssm.gate_proj * x_norm
    
    println("\n2. Projections:")
    println("   qkv norm: ", round(sqrt(sum(abs2.(qkv))), digits=3))
    println("   z norm: ", round(sqrt(sum(abs2.(z))), digits=3))
    println("   z sample: ", z[1:5])
    
    # Step 3: Convolution
    if ssm.conv_kernel > 1
        ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
    end
    ssm.conv_state[:, ssm.conv_kernel] .= qkv
    
    x_conv = Vector{Float32}(undef, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
    end
    
    println("\n3. After convolution (before SiLU):")
    println("   x_conv norm: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    
    # Step 4: SiLU on x_conv
    x_conv_silu = x_conv .* (1.0f0 ./ (1.0f0 .+ exp.(-x_conv)))
    println("\n4. After SiLU on x_conv:")
    println("   x_conv norm: ", round(sqrt(sum(abs2.(x_conv_silu))), digits=3))
    
    # Step 5: Delta net processing (simplified - just check output)
    # Split Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv_silu, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv_silu, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv_silu, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    println("\n5. Q, K, V after SiLU:")
    println("   Q norm: ", round(sqrt(sum(abs2.(q_all))), digits=3))
    println("   K norm: ", round(sqrt(sum(abs2.(k_all))), digits=3))
    println("   V norm: ", round(sqrt(sum(abs2.(v_all))), digits=3))
    
    # Alpha/beta
    alpha_proj = ssm.ssm_alpha_weight * x_norm
    beta_proj = ssm.ssm_beta_weight * x_norm
    
    println("\n6. Alpha/beta projections:")
    println("   alpha_proj sample: ", alpha_proj[1:3])
    println("   beta_proj sample: ", beta_proj[1:3])
    
    # Process heads (simplified)
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
    
    println("\n7. After delta net (before ssm_norm):")
    println("   y_all norm: ", round(sqrt(sum(abs2.(y_all))), digits=6))
    
    # Step 8: SSM norm
    for h in 1:ssm.num_v_heads
        y_h = view(y_all, (h-1)*ssm.head_v_dim+1:h*ssm.head_v_dim)
        Inferno.ModelCPU.rmsnorm_cpu!(y_h, y_h, ssm.ssm_norm)
    end
    
    println("\n8. After ssm_norm:")
    println("   y_all norm: ", round(sqrt(sum(abs2.(y_all))), digits=3))
    
    # Step 9: SiLU gate on z
    # silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
    # Output: y_all * silu(z)
    silu_z = z .* (1.0f0 ./ (1.0f0 .+ exp.(-z)))
    y_gated = y_all .* silu_z
    
    println("\n9. After SiLU gate:")
    println("   silu(z) norm: ", round(sqrt(sum(abs2.(silu_z))), digits=3))
    println("   y_gated norm: ", round(sqrt(sum(abs2.(y_gated))), digits=3))
    println("   silu(z) sample: ", silu_z[1:5])
    
    # Step 10: Output projection
    ssm_out = ssm.ssm_out * y_gated
    
    println("\n10. After output projection:")
    println("    ssm_out norm: ", round(sqrt(sum(abs2.(ssm_out))), digits=3))
    
    # Step 11: Residual
    x_after_ssm = x_orig .+ ssm_out
    
    println("\n11. After SSM residual:")
    println("    norm: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    
    # Step 12: Post norm
    x_norm2 = layer.post_norm(x_after_ssm)
    
    println("\n12. After post_norm:")
    println("    norm: ", round(sqrt(sum(abs2.(x_norm2))), digits=3))
    
    # Step 13: MLP
    mlp_out = layer.mlp(x_norm2)
    
    println("\n13. MLP output:")
    println("    norm: ", round(sqrt(sum(abs2.(mlp_out))), digits=3))
    
    # Step 14: Final residual
    x_final = x_after_ssm .+ mlp_out
    
    println("\n14. Final layer 7 output:")
    println("    norm: ", round(sqrt(sum(abs2.(x_final))), digits=3))
end

trace_gate_silu_output()
