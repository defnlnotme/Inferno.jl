#!/usr/bin/env julia
# Test RMSNorm implementation

using LinearAlgebra
using Statistics

# Our RMSNorm
function rmsnorm_ours(x, weight; eps=1e-6)
    ss = mapreduce(abs2, +, x)
    m = ss / length(x)
    scale = 1.0 / sqrt(m + eps)
    return x .* scale .* weight
end

# Reference implementation
function rmsnorm_ref(x, weight; eps=1e-6)
    rms = sqrt(mean(x.^2) + eps)
    return (x ./ rms) .* weight
end

# Test
x = randn(Float32, 256)
w = ones(Float32, 256)

y1 = rmsnorm_ours(x, w)
y2 = rmsnorm_ref(x, w)

println("Our norm: $(norm(y1))")
println("Ref norm: $(norm(y2))")
println("Max diff: $(maximum(abs.(y1 - y2)))")
println("Are they equal? $(maximum(abs.(y1 - y2)) < 1e-5)")
