using Inferno
using LinearAlgebra

function trace_layer6()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    
    # Process through first 5 layers
    for i in 1:5
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    println("=== Layer 6 (index 5) Trace ===")
    println("Input to layer 6: norm = ", round(sqrt(sum(abs2.(x))), digits=3))
    
    layer = model.layers[6]
    ssm = layer.op
    
    x_before = copy(x)
    x_norm = layer.in_norm(x)
    
    println("After in_norm: norm = ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    
    # Get projections
    qkv = ssm.in_proj * x_norm
    z = ssm.gate_proj * x_norm
    
    println("\nProjections:")
    println("  qkv norm: ", round(sqrt(sum(abs2.(qkv))), digits=3))
    println("  z norm: ", round(sqrt(sum(abs2.(z))), digits=3))
    
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
    
    println("\nAfter conv + SiLU: norm = ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    
    # Split
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    println("  Q norm: ", round(sqrt(sum(abs2.(q_all))), digits=3))
    println("  K norm: ", round(sqrt(sum(abs2.(k_all))), digits=3))
    println("  V norm: ", round(sqrt(sum(abs2.(v_all))), digits=3))
    
    # Alpha/beta
    alpha_proj = ssm.ssm_alpha_weight * x_norm
    beta_proj = ssm.ssm_beta_weight * x_norm
    
    println("\nAlpha/beta:")
    println("  alpha_proj norm: ", round(sqrt(sum(abs2.(alpha_proj))), digits=3))
    println("  beta_proj norm: ", round(sqrt(sum(abs2.(beta_proj))), digits=3))
    
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
    
    println("\nBefore ssm_norm: y_all norm = ", round(sqrt(sum(abs2.(y_all))), digits=6))
    
    # SSM norm
    for h in 1:ssm.num_v_heads
        y_h = view(y_all, (h-1)*ssm.head_v_dim+1:h*ssm.head_v_dim)
        Inferno.ModelCPU.rmsnorm_cpu!(y_h, y_h, ssm.ssm_norm)
    end
    
    println("After ssm_norm: y_all norm = ", round(sqrt(sum(abs2.(y_all))), digits=3))
    
    # SiLU gate
    silu_z = z .* (1.0f0 ./ (1.0f0 .+ exp.(-z)))
    y_gated = y_all .* silu_z
    
    println("\nAfter SiLU gate: y_gated norm = ", round(sqrt(sum(abs2.(y_gated))), digits=3))
    
    # Output projection
    ssm_output = ssm.ssm_out * y_gated
    
    println("\nSSM output (after projection): norm = ", round(sqrt(sum(abs2.(ssm_output))), digits=3))
    
    # Residual
    x_after_ssm = x_before .+ ssm_output
    println("\nAfter residual: norm = ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    
    # Post norm
    x_post = layer.post_norm(x_after_ssm)
    println("After post_norm: norm = ", round(sqrt(sum(abs2.(x_post))), digits=3))
    
    # MLP
    mlp_out = layer.mlp(x_post)
    println("MLP output: norm = ", round(sqrt(sum(abs2.(mlp_out))), digits=3))
    
    # Final
    x_final = x_after_ssm .+ mlp_out
    println("Final output: norm = ", round(sqrt(sum(abs2.(x_final))), digits=3))
    
    println("\n=== Comparison with llama.cpp ===")
    println("llama.cpp:")
    println("  final_output-5 (before ssm_out): 2.43")
    println("  attn_residual-5: 2.80")
    println("  ffn_out-5: 0.79")
    println("  l_out-5: 2.76")
    println("")
    println("Our implementation:")
    println("  y_gated (before ssm_out): $(round(sqrt(sum(abs2.(y_gated))), digits=3))")
    println("  ssm_output (after ssm_out): $(round(sqrt(sum(abs2.(ssm_output))), digits=3))")
    println("  x_after_ssm: $(round(sqrt(sum(abs2.(x_after_ssm))), digits=3))")
    println("  mlp_out: $(round(sqrt(sum(abs2.(mlp_out))), digits=3))")
    println("  x_final: $(round(sqrt(sum(abs2.(x_final))), digits=3))")
end

trace_layer6()
