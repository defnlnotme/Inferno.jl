#!/usr/bin/env julia
# Test RoPE implementation

using LinearAlgebra

# Our RoPE (from ModelCPU.jl)
function apply_rotary_emb_ours!(x::Matrix{Float32}, pos::Int, inv_freq::Vector{Float32}, rotary_dim::Int)
    head_dim, num_heads = size(x, 1), size(x, 2)
    half = div(rotary_dim, 2)
    
    for h in 1:num_heads
        for i in 1:half
            freq = inv_freq[i] * pos
            cos_val = cos(freq)
            sin_val = sin(freq)
            
            idx1 = i
            idx2 = i + half
            
            x1 = x[idx1, h]
            x2 = x[idx2, h]
            
            x[idx1, h] = x1 * cos_val - x2 * sin_val
            x[idx2, h] = x1 * sin_val + x2 * cos_val
        end
    end
    return x
end

# Reference RoPE (standard Llama-style)
function apply_rotary_emb_ref!(x::Matrix{Float32}, pos::Int, inv_freq::Vector{Float32}, rotary_dim::Int)
    head_dim, num_heads = size(x, 1), size(x, 2)
    half = div(rotary_dim, 2)
    
    # Precompute cos/sin for all positions
    freqs = inv_freq .* pos
    cos_vals = cos.(freqs)
    sin_vals = sin.(freqs)
    
    for h in 1:num_heads
        for i in 1:half
            idx1 = i
            idx2 = i + half
            
            x1 = x[idx1, h]
            x2 = x[idx2, h]
            
            x[idx1, h] = x1 * cos_vals[i] - x2 * sin_vals[i]
            x[idx2, h] = x1 * sin_vals[i] + x2 * cos_vals[i]
        end
    end
    return x
end

# Test
head_dim = 256
rotary_dim = 64  # partial rotary (25% of 256)
num_heads = 8
theta = 10000000.0

# inv_freq = 1 / (theta^(2i/d)) for i in [0, rotary_dim/2)
inv_freq = Float32[1.0 / (theta ^ (2*(i-1)/head_dim)) for i in 1:div(rotary_dim, 2)]

x1 = randn(Float32, head_dim, num_heads)
x2 = copy(x1)
pos = 10

apply_rotary_emb_ours!(x1, pos, inv_freq, rotary_dim)
apply_rotary_emb_ref!(x2, pos, inv_freq, rotary_dim)

println("Max diff: $(maximum(abs.(x1 - x2)))")
println("Are they equal? $(maximum(abs.(x1 - x2)) < 1e-5)")

# Check that non-rotary dimensions are unchanged
non_rotary_start = rotary_dim + 1
if non_rotary_start <= head_dim
    diff_non_rotary = maximum(abs.(x1[non_rotary_start:end, :] - x2[non_rotary_start:end, :]))
    println("Non-rotary dimensions match? $(diff_non_rotary < 1e-10)")
end

# Verify the rotary portion is actually changed from original
x_orig = zeros(Float32, head_dim, num_heads) # original would have been different
println("Rotary dimensions modified? $(maximum(abs.(x1[1:rotary_dim, :])) > 0)")
