using Inferno
using LinearAlgebra
using Statistics
using LoopVectorization

function benchmark_ssm_threading()
    GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    # Find first SSM layer
    ssm_layer_idx = findfirst(l -> l.op isa Inferno.ModelCPU.GatedDeltaNetCPU, model.layers)
    layer = model.layers[ssm_layer_idx]
    ssm = layer.op
    
    println("=== SSM Threading Benchmark ===")
    println("Num v heads: $(ssm.num_v_heads)")
    println("Num k heads: $(ssm.num_k_heads)")
    println("├ We're processing $(ssm.num_v_heads) heads sequentially")
    println("├ Each head does: decay + matmul + outer product + matmul")
    println("└ Perfect for parallelization!")
    println()
    
    # Prepare pre-computed data (typical after conv/SiLU)
    x = randn(Float32, 1024) * 0.01f0
    
    # Pre-run to populate conv state
    cache = Inferno.ModelCPU.init_kv_cache_cpu(model.config)
    qkv = ssm.qkv_buf
    z = ssm.z_buf
    mul!(qkv, ssm.in_proj, x)
    mul!(z, ssm.gate_proj, x)
    
    # Conv
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
    
    # Conv + SiLU
    x_conv = ssm.x_conv_buf
    @turbo for c in 1:ssm.conv_channels
        v = zero(Float32)
        for k in 1:ssm.conv_kernel
            v += ssm.conv_state[c, k] * ssm.ssm_conv1d[c, k]
        end
        x_conv[c] = v / (1.0f0 + exp(-v))
    end
    
    # Alpha/beta projections (pre-compute)
    mul!(ssm.alpha_proj_buf, ssm.ssm_alpha_weight', x)
    mul!(ssm.beta_proj_buf, ssm.ssm_beta_weight', x)
    
    # Setup for head processing
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    alpha_proj = ssm.alpha_proj_buf
    beta_proj = ssm.beta_proj_buf
    
    println("Baseline: Sequential head processing")
    
    function process_heads_sequential()
        y_all = ssm.y_all_buf
        fill!(y_all, 0.0f0)
        
        eps = 1.0f-6
        
        for h in 1:ssm.num_v_heads
            g = ((h - 1) % ssm.num_k_heads) + 1
            
            qg = view(q_all, :, g)
            kg = view(k_all, :, g)
            vg = view(v_all, :, h)
            
            scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))
            
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
            
            alpha_val = clamp(Float64(alpha_proj[h]) + Float64(ssm.ssm_dt_bias[h]), -20.0, 20.0)
            softplus_alpha = log(1.0 + exp(alpha_val))
            decay = Float32(ssm.ssm_a[h] * softplus_alpha)
            decay_to_apply = Float32(exp(decay))
            
            beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
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
    
    # Warmup
    for _ in 1:10
        Inferno.ModelCPU.reset_states_cpu!(ssm)
        process_heads_sequential()
    end
    
    # Benchmark sequential
    n_iters = 200
    times_seq = Float64[]
    for _ in 1:n_iters
        Inferno.ModelCPU.reset_states_cpu!(ssm)
        t0 = time()
        process_heads_sequential()
        t1 = time()
        push!(times_seq, t1 - t0)
    end
    
    println("  Sequential: $(round(mean(times_seq) * 1000, digits=3)) ms avg")
    println("  Throughput: $(round(1/mean(times_seq), digits=1)) iters/sec")
    println()
    
    # Test: Threaded per-head processing
    println("Threaded per-head processing (with @threads)")
    
    # Need thread-local buffers - can't share
    function process_heads_threaded()
        y_all = ssm.y_all_buf
        fill!(y_all, 0.0f0)
        
        eps = 1.0f-6
        
        Threads.@threads for h in 1:ssm.num_v_heads
            # Each thread needs its own q_norm_buf, k_norm_buf, sk_buf, d_buf, y_h_buf
            # But these are shared... this won't work without duplication
            # For now, just measure overhead
            g = ((h - 1) % ssm.num_k_heads) + 1
            
            # ... rest of processing would need thread-local buffers
        end
        return y_all
    end
    
    println("  Skipped: Shared buffers don't work with @threads")
    println()
    
    # Test: Manual SIMD tuning - unroll loops
    println("SIMD tuning experiments:")
    
    # Test manual vs @turbo
    state = view(ssm.h, :, :, 1)
    k_norm = rand(Float32, ssm.head_k_dim)
    q_norm = rand(Float32, ssm.head_k_dim)
    
    # Test decay operation
    t0 = time()
    for _ in 1:1000
        @turbo state .*= 0.99f0
    end
    t1 = time()
    println("  @turbo decay: $(round((t1-t0), digits=4)) ms")
    
    t0 = time()
    for _ in 1:1000
        @inbounds @simd for i in eachindex(state)
            state[i] *= 0.99f0
        end
    end
    t1 = time()
    println("  @simd decay: $(round((t1-t0), digits=4)) ms")
    
    t0 = time()
    for _ in 1:1000
        state .*= 0.99f0
    end
    t1 = time()
    println("  broadcast decay: $(round((t1-t0), digits=4)) ms")
    println()
    
    # Test matmul: state * k
    sk = ssm.sk_buf
    state_slice = view(ssm.h, :, :, 1)
    
    t0 = time()
    for _ in 1:1000
        @turbo for i in 1:ssm.head_v_dim
            s = zero(Float32)
            for j in 1:ssm.head_k_dim
                s += state_slice[i, j] * k_norm[j]
            end
            sk[i] = s
        end
    end
    t1 = time()
    turbo_time = t1 - t0
    println("  @turbo matmul: $(round(turbo_time, digits=4)) ms")
    
    # Test with BLAS (convert to matrix form)
    # state is (head_v_dim, head_k_dim), k is (head_k_dim,)
    # sk = state * k
    t0 = time()
    for _ in 1:1000
        mul!(sk, state_slice, k_norm)
    end
    t1 = time()
    blas_time = t1 - t0
    println("  BLAS matmul: $(round(blas_time, digits=4)) ms")
    println("  Speedup: $(round(blas_time/turbo_time, digits=2))x")
end

benchmark_ssm_threading()
