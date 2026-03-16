using BenchmarkTools
using oneAPI

function init_kv_cache_baseline(head_dim, n_kv, max_seq)
    seq = min(max_seq, 512)
    # Simulate the old zeros-based implementation
    k = oneArray(zeros(Float32, head_dim, n_kv, seq))
    v = oneArray(zeros(Float32, head_dim, n_kv, seq))
    oneAPI.synchronize()
    return k, v
end

function init_kv_cache_optimized(head_dim, n_kv, max_seq)
    seq = min(max_seq, 512)
    k = oneArray{Float32}(undef, head_dim, n_kv, seq)
    v = oneArray{Float32}(undef, head_dim, n_kv, seq)
    fill!(k, 0.0f0)
    fill!(v, 0.0f0)
    oneAPI.synchronize()
    return k, v
end

println("Benchmarking baseline (zeros)...")
@btime init_kv_cache_baseline(256, 2, 4096)

println("Benchmarking optimized (fill!)...")
@btime init_kv_cache_optimized(256, 2, 4096)
