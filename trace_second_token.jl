using Inferno
using LinearAlgebra

function trace_second_token()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # First token
    tok1 = 761  # "The"
    pos1 = 0
    x1 = model.embed[:, tok1]
    
    println("=== First Token (The) ===")
    println("Embedding norm: ", round(sqrt(sum(abs2.(x1))), digits=3))
    
    for i in 1:model.config.num_hidden_layers
        x1 = model.layers[i](x1, pos1, model.rope, caches[i])
    end
    
    println("After all layers norm: ", round(sqrt(sum(abs2.(x1))), digits=3))
    
    # Second token
    tok2 = 7922  # "capital" (assuming this is in vocab)
    pos2 = 1
    x2 = model.embed[:, tok2]
    
    println("\n=== Second Token ===")
    println("Embedding norm: ", round(sqrt(sum(abs2.(x2))), digits=3))
    
    # Check layer 7 state before processing second token
    layer7 = model.layers[7]
    ssm7 = layer7.op
    
    println("\nConv state before layer 7 (second token):")
    println("  conv_state norm: ", round(sqrt(sum(abs2.(ssm7.conv_state))), digits=3))
    
    println("\nDelta net state before layer 7 (second token):")
    println("  h norm: ", round(sqrt(sum(abs2.(ssm7.h))), digits=3))
    
    # Process through layers 1-6
    for i in 1:6
        x2 = model.layers[i](x2, pos2, model.rope, caches[i])
    end
    
    println("\nAfter layers 1-6 norm: ", round(sqrt(sum(abs2.(x2))), digits=3))
    
    # Process layer 7 step by step
    x_orig = copy(x2)
    x_norm = layer7.in_norm(x2)
    
    qkv = ssm7.in_proj * x_norm
    z = ssm7.gate_proj * x_norm
    
    # Convolution
    if ssm7.conv_kernel > 1
        ssm7.conv_state[:, 1:(ssm7.conv_kernel-1)] .= ssm7.conv_state[:, 2:ssm7.conv_kernel]
    end
    ssm7.conv_state[:, ssm7.conv_kernel] .= qkv
    
    x_conv = Vector{Float32}(undef, ssm7.conv_channels)
    for c in 1:ssm7.conv_channels
        x_conv[c] = dot(view(ssm7.conv_state, c, :), view(ssm7.ssm_conv1d, :, c))
    end
    
    # SiLU
    x_conv .= x_conv .* (1.0f0 ./ (1.0f0 .+ exp.(-x_conv)))
    
    # Split into Q, K, V
    qk_size = ssm7.head_k_dim * ssm7.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm7.head_k_dim, ssm7.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm7.head_k_dim, ssm7.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm7.d_inner), ssm7.head_v_dim, ssm7.num_v_heads)
    
    println("\nAfter conv + SiLU:")
    println("  x_conv norm: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    println("  Q norm: ", round(sqrt(sum(abs2.(q_all))), digits=3))
    println("  K norm: ", round(sqrt(sum(abs2.(k_all))), digits=3))
    println("  V norm: ", round(sqrt(sum(abs2.(v_all))), digits=3))
    
    # Check alpha/beta
    alpha_proj = ssm7.ssm_alpha_weight * x_norm
    beta_proj = ssm7.ssm_beta_weight * x_norm
    
    # Process delta net for each head
    y_all = zeros(Float32, ssm7.d_inner)
    scale = 1.0f0 / sqrt(Float32(ssm7.head_k_dim))
    
    println("\nPer-head output (second token):")
    for h in 1:ssm7.num_v_heads
        g = ((h - 1) % ssm7.num_k_heads) + 1
        qg = view(q_all, :, g)
        kg = view(k_all, :, g)
        vg = view(v_all, :, h)
        
        q_norm = sqrt(sum(abs2, qg) + ssm7.ssm_norm.eps)
        k_norm = sqrt(sum(abs2, kg) + ssm7.ssm_norm.eps)
        
        q_normalized = qg ./ q_norm .* scale
        k_normalized = kg ./ k_norm
        
        alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm7.ssm_dt_bias[h]), -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        decay = Float32(exp(softplus_alpha * Float64(ssm7.ssm_a[h])))
        
        beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
        beta = Float32(1.0 / (1.0 + exp(-beta_val)))
        
        state = view(ssm7.h, :, :, h)
        state .*= decay
        
        sk = k_normalized' * state
        d = beta .* (vg .- vec(sk))
        BLAS.ger!(1.0f0, k_normalized, d, state)
        
        yg = view(y_all, (h-1)*ssm7.head_v_dim+1:h*ssm7.head_v_dim)
        mul!(yg, state', q_normalized)
        
        y_h_norm = sqrt(sum(abs2.(yg)))
        if h <= 5 || h >= 12
            println("  Head $h: yg norm = $y_h_norm")
        end
    end
    
    println("\ny_all (before norm) norm: ", round(sqrt(sum(abs2.(y_all))), digits=3))
    
    # SSM norm
    for h in 1:ssm7.num_v_heads
        y_h = view(y_all, (h-1)*ssm7.head_v_dim+1:h*ssm7.head_v_dim)
        Inferno.ModelCPU.rmsnorm_cpu!(y_h, y_h, ssm7.ssm_norm)
    end
    
    println("y_all (after norm) norm: ", round(sqrt(sum(abs2.(y_all))), digits=3))
end

trace_second_token()
