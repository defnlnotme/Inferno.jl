#!/usr/bin/env julia
# flash_attention_bench.jl - Compare standard vs Flash Attention
# Usage: julia --project=. benchmark/flash_attention_bench.jl

using BenchmarkTools, Statistics
using LinearAlgebra
using Printf

include("../src/FlashAttention.jl")

"""
Time flash attention for a single head at different sequence lengths.
"""
function benchmark_flash_attention(seq_len::Int, n_trials::Int=100)
    head_dim = 256
    n_kv_heads = 2
    block_size = 64

    # Setup
    Q = rand(Float32, head_dim)
    cache_k = rand(Float32, head_dim, n_kv_heads, seq_len)
    cache_v = rand(Float32, head_dim, n_kv_heads, seq_len)
    output = Vector{Float32}(undef, head_dim)
    scale = Float32(1.0 / sqrt(head_dim))

    # Warmup
    flash_attention_cpu!(output, Q, cache_k, cache_v, 1, seq_len, scale, head_dim)

    # Benchmark
    times = Float64[]
    for _ in 1:n_trials
        t = @elapsed flash_attention_cpu!(output, Q, cache_k, cache_v, 1, seq_len, scale, head_dim)
        push!(times, t * 1000) # Convert to ms
    end

    return times
end

"""
Time standard attention (naive O(N²) implementation)
"""
function benchmark_standard_attention(seq_len::Int, n_trials::Int=100)
    head_dim = 256
    n_kv_heads = 2
    kv_h = 1

    # Setup
    Q = rand(Float32, head_dim)
    cache_k = rand(Float32, head_dim, n_kv_heads, seq_len)
    cache_v = rand(Float32, head_dim, n_kv_heads, seq_len)
    scale = Float32(1.0 / sqrt(head_dim))

    # Warmup
    scores = similar(Q, seq_len)
    output = similar(Q)

    # Get K and V for this head
    K_h = view(cache_k, :, kv_h, 1:seq_len)
    V_h = view(cache_v, :, kv_h, 1:seq_len)

    # Benchmark
    times = Float64[]
    for _ in 1:n_trials
        t = @elapsed begin
            # Compute attention scores: K^T @ Q
            for i in 1:seq_len
                s = zero(Float32)
                for j in 1:head_dim
                    s += K_h[j, i] * Q[j]
                end
                scores[i] = s * scale
            end

            # Softmax
            max_score = maximum(scores)
            for i in 1:seq_len
                scores[i] = exp(scores[i] - max_score)
            end
            sum_scores = sum(scores)
            for i in 1:seq_len
                scores[i] /= sum_scores
            end

            # Weighted sum: V @ scores
            for i in 1:head_dim
                s = zero(Float32)
                for j in 1:seq_len
                    s += V_h[i, j] * scores[j]
                end
                output[i] = s
            end
        end
        push!(times, t * 1000) # Convert to ms
    end

    return times
end

println("="^70)
println("Flash Attention vs Standard Attention Benchmark")
println("="^70)
println("\nComparing memory-efficient Flash Attention vs naive O(N²) implementation")
println()

# Test at various sequence lengths
seq_lens = [64, 128, 256, 512, 1024, 2048]
n_trials = 100

print_results = []

for seq_len in seq_lens
    # Standard attention
    std_times = benchmark_standard_attention(seq_len, n_trials)
    std_mean = mean(std_times)
    std_median = median(std_times)

    # Flash attention
    fa_times = benchmark_flash_attention(seq_len, n_trials)
    fa_mean = mean(fa_times)
    fa_median = median(fa_times)

    speedup = std_mean / fa_mean

    row = (seq_len, std_mean, std_median, fa_mean, fa_median, speedup)
    push!(print_results, row)

    @printf "Seq_len=%4d | Standard: %7.3f ±%6.3f ms | Flash: %7.3f ±%6.3f ms | Speedup: %.2fx\n" seq_len std_mean std(std_times) fa_mean std(fa_times) speedup
end

println("\n" * "="^70)
println("Summary: Flash Attention maintains consistent memory usage")
println("Both implementations are O(N) time per token on CPU (Flash memory, not speed)")
println("Flash Attention shows benefits mainly for:")
println("  1. Lower memory bandwidth (fewer cache misses)")
println("  2. Better numerical stability (online softmax)")
println("="^70)

# Save results to file
open("flash_attention_results.json", "w") do f
    println(f, "[")
    for (i, row) in enumerate(print_results)
        seq_len, std_mean, std_median, fa_mean, fa_median, speedup = row
        println(f, "  {\"seq_len\": $seq_len, \"standard_ms\": $std_mean, \"flash_ms\": $fa_mean, \"speedup\": $speedup}", i < length(print_results) ? "," : "")
    end
    println(f, "]")
end
println("\nResults saved to flash_attention_results.json")
