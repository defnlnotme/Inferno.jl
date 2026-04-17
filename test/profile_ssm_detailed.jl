using Inferno
using LinearAlgebra
using Statistics

function profile_ssm_detailed()
    GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    # Find first SSM layer - get the op from the DecoderLayerCPU
    ssm_layer_idx = findfirst(l -> l.op isa Inferno.ModelCPU.GatedDeltaNetCPU, model.layers)
    layer = model.layers[ssm_layer_idx]
    ssm_layer = layer.op
    
    println("=== SSM Detailed Profiling (Layer $ssm_layer_idx) ===")
    println("Config: num_v_heads=$(ssm_layer.num_v_heads), num_k_heads=$(ssm_layer.num_k_heads)")
    println("        head_k_dim=$(ssm_layer.head_k_dim), head_v_dim=$(ssm_layer.head_v_dim)")
    println("        d_inner=$(ssm_layer.d_inner), conv_channels=$(ssm_layer.conv_channels)")
    println()
    
    # Create test input
    x = randn(Float32, 1024) * 0.01f0
    cache = Inferno.ModelCPU.init_kv_cache_cpu(model.config)
    
    # Warm up
    for _ in 1:10
        Inferno.ModelCPU.reset_states_cpu!(ssm_layer)
        _ = ssm_layer(x, 1, model.rope, cache)
    end
    
    # Timed run
    n_iters = 100
    times = Float64[]
    
    for _ in 1:n_iters
        Inferno.ModelCPU.reset_states_cpu!(ssm_layer)
        t0 = time()
        _ = ssm_layer(x, 1, model.rope, cache)
        t1 = time()
        push!(times, t1 - t0)
    end
    
    avg_time = mean(times)
    min_time = minimum(times)
    med_time = median(times)
    
    println("SSM forward pass timing ($(n_iters) iters):")
    println("  Mean: $(round(avg_time * 1000, digits=2)) ms")
    println("  Min:  $(round(min_time * 1000, digits=2)) ms")
    println("  Med:  $(round(med_time * 1000, digits=2)) ms")
    println("  Throughput: $(round(1/avg_time, digits=1)) layers/sec")
    println()
    
    # Calculate projected times for full model
    num_ssm = count(l -> l.op isa Inferno.ModelCPU.GatedDeltaNetCPU, model.layers)
    num_attn = count(l -> l.op isa Inferno.ModelCPU.FullAttentionCPU, model.layers)
    
    println("Model composition:")
    println("  SSM layers: $num_ssm")
    println("  Attn layers: $num_attn")
    println("  Total layers: $(length(model.layers))")
    println()
    println("Projected per-token time:")
    println("  SSM layers: $(round(avg_time * 1000 * num_ssm, digits=1)) ms")
    println("  (Assuming attn = 1.9ms each) Attn layers: $(round(1.9 * num_attn, digits=1)) ms")
    println("  Estimated total: $(round(avg_time * 1000 * num_ssm + 1.9 * num_attn, digits=1)) ms")
end

profile_ssm_detailed()

println()
println("=== Quick BLAS threading test ===")
println("Current BLAS threads: ", BLAS.get_num_threads())

# Test if we can control BLAS threading level
for nt in [1, 2, 4, 8]
    BLAS.set_num_threads(nt)
    println("BLAS threads set to $nt: actual = ", BLAS.get_num_threads())
end
