using Inferno
using LinearAlgebra

function trace_ssm_step_by_step()
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
    
    println("=== SSM Step-by-Step for Layer 7, First Token ===")
    println("Input x norm: ", round(sqrt(sum(abs2.(x))), digits=3))
    
    layer = model.layers[7]
    ssm = layer.op
    
    # Get normalized input
    x_norm = layer.in_norm(x)
    println("After in_norm: ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    
    # 1. Input projections
    qkv = ssm.in_proj * x_norm
    z = ssm.gate_proj * x_norm
    
    println("\n=== Projections ===")
    println("qkv norm: ", round(sqrt(sum(abs2.(qkv))), digits=3))
    println("z norm: ", round(sqrt(sum(abs2.(z))), digits=3))
    println("z sample: ", z[1:5])
    
    # 2. Update conv state
    if ssm.conv_kernel > 1
        ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
    end
    ssm.conv_state[:, ssm.conv_kernel] .= qkv
    
    println("\n=== After conv state update ===")
    println("conv_state norm: ", round(sqrt(sum(abs2.(ssm.conv_state))), digits=3))
    
    # 3. Compute convolution
    x_conv = Vector{Float32}(undef, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
    end
    
    println("\n=== After convolution ===")
    println("x_conv norm before SiLU: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    println("x_conv sample before SiLU: ", x_conv[1:5])
    
    # 4. SiLU activation
    @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))
    
    println("x_conv norm after SiLU: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    println("x_conv sample after SiLU: ", x_conv[1:5])
    
    # 5. Split into Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    println("\n=== Q, K, V splits ===")
    println("Q norm: ", round(sqrt(sum(abs2.(q_all))), digits=3))
    println("K norm: ", round(sqrt(sum(abs2.(k_all))), digits=3))
    println("V norm: ", round(sqrt(sum(abs2.(v_all))), digits=3))
    
    # 6. Alpha/beta projections
    alpha_proj = ssm.ssm_alpha_weight * x_norm
    beta_proj = ssm.ssm_beta_weight * x_norm
    
    println("\n=== Alpha/Beta projections ===")
    println("alpha_proj: ", alpha_proj[1:min(5, length(alpha_proj))])
    println("beta_proj: ", beta_proj[1:min(5, length(beta_proj))])
    
    # 7. Process each head
    println("\n=== Per-head processing ===")
    y_all = zeros(Float32, ssm.d_inner)
    scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
    
    for h in 1:ssm.num_v_heads
        g = ((h - 1) % ssm.num_k_heads) + 1
        
        qg = view(q_all, :, g)
        kg = view(k_all, :, g)
        vg = view(v_all, :, h)
        
        # Q/K L2 normalization
        q_norm = sqrt(sum(abs2, qg) + ssm.ssm_norm.eps)
        k_norm = sqrt(sum(abs2, kg) + ssm.ssm_norm.eps)
        
        q_normalized = qg ./ q_norm .* scale
        k_normalized = kg ./ k_norm
        
        # Gate values
        alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        softplus_alpha = clamp(softplus_alpha, -10.0, 10.0)
        
        decay = Float32(exp(softplus_alpha * Float64(ssm.ssm_a[h])))
        decay = clamp(decay, 0.0f0, 1.0f0)
        
        beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
        beta = Float32(1.0 / (1.0 + exp(-beta_val)))
        
        if h <= 3
            println("\nHead $h:")
            println("  q_norm raw: ", round(sqrt(sum(abs2, qg)), digits=3))
            println("  k_norm raw: ", round(sqrt(sum(abs2, kg)), digits=3))
            println("  v_norm raw: ", round(sqrt(sum(abs2, vg)), digits=3))
            println("  alpha_val: ", round(alpha_val, digits=3))
            println("  softplus_alpha: ", round(softplus_alpha, digits=3))
            println("  ssm_a[h]: ", ssm.ssm_a[h])
            println("  decay: ", round(decay, digits=6))
            println("  beta: ", round(beta, digits=3))
        end
        
        # State operations
        state = view(ssm.h, :, :, h)
        state .*= decay
        
        # sk = k' * state
        sk = k_normalized' * state
        
        # d = beta * (v - sk)
        d = beta .* (vg .- vec(sk))
        
        # state += k * d'
        BLAS.ger!(1.0f0, k_normalized, d, state)
        
        # Output: o = q' * state
        yg = view(y_all, (h-1)*ssm.head_v_dim+1:h*ssm.head_v_dim)
        mul!(yg, state', q_normalized)
        
        if h <= 3
            println("  state norm after update: ", round(sqrt(sum(abs2.(state))), digits=3))
            println("  yg norm: ", round(sqrt(sum(abs2.(yg))), digits=3))
        end
    end
    
    println("\n=== After head processing ===")
    println("y_all norm before ssm_norm: ", round(sqrt(sum(abs2.(y_all))), digits=3))
    
    # 8. Apply SSM norm (per-head normalization)
    for h in 1:ssm.num_v_heads
        y_h = view(y_all, (h-1)*ssm.head_v_dim+1:h*ssm.head_v_dim)
        Inferno.ModelCPU.rmsnorm_cpu!(y_h, y_h, ssm.ssm_norm)
    end
    
    println("y_all norm after ssm_norm: ", round(sqrt(sum(abs2.(y_all))), digits=3))
    
    # 9. SiLU gate on z
    @. y_all = y_all * z * (1.0f0 / (1.0f0 + exp(-z)))
    
    println("y_all norm after SiLU gate: ", round(sqrt(sum(abs2.(y_all))), digits=3))
    
    # 10. Output projection
    output = ssm.ssm_out * y_all
    
    println("\n=== Final SSM output ===")
    println("output norm: ", round(sqrt(sum(abs2.(output))), digits=3))
end

trace_ssm_step_by_step()
