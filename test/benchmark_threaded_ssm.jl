using Inferno
using LinearAlgebra
using Statistics
using LoopVectorization

"""
Thread-parallel SSM head processing with local buffers.

Each thread gets its own local work buffers to avoid false sharing.
"""
function process_heads_threaded!(m::Inferno.ModelCPU.GatedDeltaNetCPU, 
                                  q_all, k_all, v_all, alpha_proj, beta_proj)
    num_heads = m.num_v_heads
    eps = 1.0f-6
    scale = 1.0f0 / sqrt(Float32(m.head_k_dim))
    
    # Each thread processes heads independently with local buffers
    Threads.@threads for h in 1:num_heads
        g = ((h - 1) % m.num_k_heads) + 1
        
        qg = view(q_all, :, g)
        kg = view(k_all, :, g)
        vg = view(v_all, :, h)
        
        # Use thread-local buffers (stack allocated or task-local)
        # For thread safety, we use local stack arrays
        q_norm = zeros(Float32, m.head_k_dim)
        k_norm = zeros(Float32, m.head_k_dim)
        sk = zeros(Float32, m.head_v_dim)
        d = zeros(Float32, m.head_v_dim)
        y_h = zeros(Float32, m.head_v_dim)
        
        # L2 normalize
        q_sum_sq = 0.0f0
        k_sum_sq = 0.0f0
        @inbounds @simd ivdep for i in 1:m.head_k_dim
            q_sum_sq += qg[i] * qg[i]
            k_sum_sq += kg[i] * kg[i]
        end
        
        q_norm_val = 1.0f0 / (sqrt(q_sum_sq) + eps) * scale
        k_norm_val = 1.0f0 / (sqrt(k_sum_sq) + eps)
        
        @inbounds @simd ivdep for i in 1:m.head_k_dim
            q_norm[i] = qg[i] * q_norm_val
            k_norm[i] = kg[i] * k_norm_val
        end
        
        # Gate values
        alpha_val = clamp(Float64(alpha_proj[h]) + Float64(m.ssm_dt_bias[h]), -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        decay = Float32(m.ssm_a[h] * softplus_alpha)
        decay_to_apply = Float32(exp(decay))
        
        beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
        beta_gate = Float32(1.0 / (1.0 + exp(-beta_val)))
        
        # State view
        state = view(m.h, :, :, h)
        
        # Decay with @simd (safe on single-threaded view)
        @inbounds @simd for i in eachindex(state)
            state[i] *= decay_to_apply
        end
        
        # sk = state * k
        @turbo for i in 1:m.head_v_dim
            s = zero(Float32)
            for j in 1:m.head_k_dim
                s += state[i, j] * k_norm[j]
            end
            sk[i] = s
        end
        
        # d = beta * (v - sk)
        @turbo for i in 1:m.head_v_dim
            d[i] = beta_gate * (vg[i] - sk[i])
        end
        
        # State update: outer product
        BLAS.ger!(1.0f0, d, k_norm, state)
        
        # y = state * q
        @turbo for i in 1:m.head_v_dim
            s = zero(Float32)
            for j in 1:m.head_k_dim
                s += state[i, j] * q_norm[j]
            end
            y_h[i] = s
        end
        
        # Write output
        yg = view(m.y_all_buf, (h-1)*m.head_v_dim+1:h*m.head_v_dim)
        yg .= y_h
    end
    
    return m.y_all_buf
end

function benchmark_threaded_vs_serial()
    GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    ssm_layer_idx = findfirst(l -> l.op isa Inferno.ModelCPU.GatedDeltaNetCPU, model.layers)
    layer = model.layers[ssm_layer_idx]
    ssm = layer.op
    
    println("=== Threaded vs Serial SSM Benchmark ===")
    println("Julia threads: ", Threads.nthreads())
    println("BLAS threads: ", BLAS.get_num_threads())
    println()
    
    # Prepare data
    x = randn(Float32, 1024) * 0.01f0
    cache = Inferno.ModelCPU.init_kv_cache_cpu(model.config)
    
    # Run once through to set up state
    qkv = ssm.qkv_buf
    z = ssm.z_buf
    mul!(qkv, ssm.in_proj, x)
    mul!(z, ssm.gate_proj, x)
    
    if ssm.conv_kernel > 1
        for j in 1:(ssm.conv_kernel-1)
            @simd ivdep for i in 1:ssm.conv_channels
                ssm.conv_state[i, j] = ssm.conv_state[i, j+1]
            end
        end
    end
    @simd ivdep for i in 1:ssm.conv_channels
        ssm.conv_state[i, ssm.conv_kernel] = qkv[i]
    end
    
    x_conv = ssm.x_conv_buf
    @turbo for c in 1:ssm.conv_channels
        v = zero(Float32)
        for k in 1:ssm.conv_kernel
            v += ssm.conv_state[c, k] * ssm.ssm_conv1d[c, k]
        end
        x_conv[c] = v / (1.0f0 + exp(-v))
    end
    
    mul!(ssm.alpha_proj_buf, ssm.ssm_alpha_weight', x)
    mul!(ssm.beta_proj_buf, ssm.ssm_beta_weight', x)
    
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    # Warmup
    for _ in 1:20
        Inferno.ModelCPU.reset_states_cpu!(ssm)
        process_heads_threaded!(ssm, q_all, k_all, v_all, ssm.alpha_proj_buf, ssm.beta_proj_buf)
    end
    
    # Benchmark threaded
    n_iters = 100
    
    # Reduce BLAS threads to avoid oversubscription
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    
    t0 = time()
    for _ in 1:n_iters
        Inferno.ModelCPU.reset_states_cpu!(ssm)
        process_heads_threaded!(ssm, q_all, k_all, v_all, ssm.alpha_proj_buf, ssm.beta_proj_buf)
    end
    t1 = time()
    threaded_time = (t1 - t0) / n_iters
    
    println("Threaded (BLAS=1): $(round(threaded_time * 1000, digits=3)) ms avg")
    println("  Throughput: $(round(1/threaded_time, digits=1)) iters/sec")
    println()
    
    # Benchmark serial
    function process_heads_serial!()
        y_all = ssm.y_all_buf
        fill!(y_all, 0.0f0)
        
        eps = 1.0f-6
        scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
        
        for h in 1:ssm.num_v_heads
            g = ((h - 1) % ssm.num_k_heads) + 1
            
            qg = view(q_all, :, g)
            kg = view(k_all, :, g)
            vg = view(v_all, :, h)
            
            # L2 normalize
            q_sum_sq = 0.0f0
            k_sum_sq = 0.0f0
            @simd ivdep for i in 1:ssm.head_k_dim
                q_sum_sq += qg[i] * qg[i]
                k_sum_sq += kg[i] * kg[i]
            end
            
            q_norm_val = 1.0f0 / (sqrt(q_sum_sq) + eps) * scale
            k_norm_val = 1.0f0 / (sqrt(k_sum_sq) + eps)
            
            @simd ivdep for i in 1:ssm.head_k_dim
                ssm.q_norm_buf[i] = qg[i] * q_norm_val
                ssm.k_norm_buf[i] = kg[i] * k_norm_val
            end
            
            q_normalized = ssm.q_norm_buf
            k_normalized = ssm.k_norm_buf
            
            alpha_val = clamp(Float64(ssm.alpha_proj_buf[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
            softplus_alpha = log(1.0 + exp(alpha_val))
            decay = Float32(ssm.ssm_a[h] * softplus_alpha)
            decay_to_apply = Float32(exp(decay))
            
            beta_val = clamp(Float64(ssm.beta_proj_buf[h]), -20.0, 20.0)
            beta_gate = Float32(1.0 / (1.0 + exp(-beta_val)))
            
            state = view(ssm.h, :, :, h)
            @turbo state .*= decay_to_apply
            
            sk = ssm.sk_buf
            @turbo for i in 1:ssm.head_v_dim
                s = zero(Float32)
                for j in 1:ssm.head_k_dim
                    s += state[i, j] * k_normalized[j]
                end
                sk[i] = s
            end
            
            d = ssm.d_buf
            @turbo @. d = beta_gate * (vg - sk)
            
            BLAS.ger!(1.0f0, d, k_normalized, state)
            
            y_h = ssm.y_h_buf
            @turbo for i in 1:ssm.head_v_dim
                s = zero(Float32)
                for j in 1:ssm.head_k_dim
                    s += state[i, j] * q_normalized[j]
                end
                y_h[i] = s
            end
            
            yg = view(y_all, (h-1)*ssm.head_v_dim+1:h*ssm.head_v_dim)
            yg .= y_h
        end
        return y_all
    end
    
    # Warmup serial
    for _ in 1:20
        Inferno.ModelCPU.reset_states_cpu!(ssm)
        process_heads_serial!()
    end
    
    t0 = time()
    for _ in 1:n_iters
        Inferno.ModelCPU.reset_states_cpu!(ssm)
        process_heads_serial!()
    end
    t1 = time()
    serial_time = (t1 - t0) / n_iters
    
    println("Serial (BLAS=$old_blas_threads): $(round(serial_time * 1000, digits=3)) ms avg")
    println("  Throughput: $(round(1/serial_time, digits=1)) iters/sec")
    println()
    
    speedup = serial_time / threaded_time
    println("Speedup: $(round(speedup, digits=2))x")
    println("Efficiency: $(round(speedup/Threads.nthreads()*100, digits=1))% (with $(Threads.nthreads()) threads)")
    
    # Restore BLAS threads
    BLAS.set_num_threads(old_blas_threads)
end

benchmark_threaded_vs_serial()
