
using BenchmarkTools
using LoopVectorization
using Statistics

# Dummy structs to match the signature
struct RMSNormCPU
    weight::Vector{Float32}
    eps::Float32
end

struct RotaryEmbeddingCPU
    inv_freq::Vector{Float32}
    max_seq_len::Int
    rotary_dim::Int
    cos_cache::Matrix{Float32}
    sin_cache::Matrix{Float32}
end

# Current implementation
function rmsnorm_rotary_old!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU, norm::RMSNormCPU)
    head_dim, num_heads = size(x, 1), size(x, 2)
    half = div(rope.rotary_dim, 2)
    weight = norm.weight

    pos_idx = pos + 1
    @inbounds begin
        cos_vals = view(rope.cos_cache, :, pos_idx)
        sin_vals = view(rope.sin_cache, :, pos_idx)
    end

    for h in 1:num_heads
        sum_sq = 0.0f0
        for i in 1:head_dim
            @inbounds sum_sq += x[i, h] * x[i, h]
        end
        rms = sqrt(sum_sq / head_dim + norm.eps)
        inv_rms = 1.0f0 / rms

        for i in 1:head_dim
            @inbounds x[i, h] = x[i, h] * inv_rms * weight[i]
        end

        for i in 1:half
            @inbounds begin
                idx1 = i
                idx2 = i + half
                x1 = x[idx1, h]
                x2 = x[idx2, h]
                c = cos_vals[i]
                s = sin_vals[i]
                x[idx1, h] = x1 * c - x2 * s
                x[idx2, h] = x1 * s + x2 * c
            end
        end
    end
    return x
end

# Optimized implementation
function rmsnorm_rotary_new!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU, norm::RMSNormCPU)
    head_dim, num_heads = size(x, 1), size(x, 2)
    half = div(rope.rotary_dim, 2)
    weight = norm.weight
    eps = norm.eps

    pos_idx = pos + 1
    @inbounds begin
        cos_vals = view(rope.cos_cache, :, pos_idx)
        sin_vals = view(rope.sin_cache, :, pos_idx)
    end

    for h in 1:num_heads
        sum_sq = 0.0f0
        @turbo for i in 1:head_dim
            sum_sq += x[i, h] * x[i, h]
        end
        inv_rms = 1.0f0 / sqrt(sum_sq / head_dim + eps)

        # Combined normalization and RoPE
        @turbo for i in 1:half
            idx1 = i
            idx2 = i + half

            x1 = x[idx1, h] * inv_rms * weight[idx1]
            x2 = x[idx2, h] * inv_rms * weight[idx2]

            c = cos_vals[i]
            s = sin_vals[i]

            x[idx1, h] = x1 * c - x2 * s
            x[idx2, h] = x1 * s + x2 * c
        end

        # Finish normalization for remaining dimensions
        @turbo for i in (rope.rotary_dim + 1):head_dim
            x[i, h] = x[i, h] * inv_rms * weight[i]
        end
    end
    return x
end

# Setup
head_dim = 128
num_heads = 32
rotary_dim = 64
max_seq_len = 2048
pos = 100

x_orig = rand(Float32, head_dim, num_heads)
weight = rand(Float32, head_dim)
eps = 1e-6f0
norm = RMSNormCPU(weight, eps)

inv_freq = rand(Float32, div(rotary_dim, 2))
cos_cache = rand(Float32, div(rotary_dim, 2), max_seq_len)
sin_cache = rand(Float32, div(rotary_dim, 2), max_seq_len)
rope = RotaryEmbeddingCPU(inv_freq, max_seq_len, rotary_dim, cos_cache, sin_cache)

# Verification
x1 = copy(x_orig)
x2 = copy(x_orig)

rmsnorm_rotary_old!(x1, pos, rope, norm)
rmsnorm_rotary_new!(x2, pos, rope, norm)

if !isapprox(x1, x2, rtol=1e-5)
    println("VERIFICATION FAILED!")
    println("Max diff: ", maximum(abs.(x1 .- x2)))
else
    println("VERIFICATION PASSED")
end

# Benchmarking
println("Benchmarking Old Implementation:")
b_old = @benchmark rmsnorm_rotary_old!(x, $pos, $rope, $norm) setup=(x=copy($x_orig))
display(b_old)
println()

println("Benchmarking New Implementation:")
b_new = @benchmark rmsnorm_rotary_new!(x, $pos, $rope, $norm) setup=(x=copy($x_orig))
display(b_new)
println()

println("Speedup: ", mean(b_old.times) / mean(b_new.times), "x")
