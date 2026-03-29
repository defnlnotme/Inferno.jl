#!/usr/bin/env julia

# Test if the issue is with weight loading or computation
using Printf

# Simple test: compute a single attention score manually
println("=== Manual Attention Debug ===")

# Simulate the computation
function test_attention_math()
    # Simple test: single head, single position
    head_dim = 256
    
    # Create random query and key
    q = randn(Float32, head_dim)
    k = randn(Float32, head_dim)
    
    # Normalize (simulate q_norm and k_norm)
    q_norm_val = sqrt(sum(q.^2) / head_dim + 1e-6f0)
    k_norm_val = sqrt(sum(k.^2) / head_dim + 1e-6f0)
    
    q_normed = q ./ q_norm_val
    k_normed = k ./ k_norm_val
    
    # Apply RoPE would happen here (skip for now)
    
    # Compute attention score
    scale = 1.0f0 / sqrt(Float32(head_dim))
    score = dot(q_normed, k_normed) * scale
    
    println("Q norm: $(norm(q_normed))")
    println("K norm: $(norm(k_normed))")
    println("Score: $score")
    println("Expected range: roughly [-scale, scale] = [-$(scale), $(scale)]")
    
    # Now test the gate application
    gate = randn(Float32, head_dim)
    output = q_normed .* sigmoid.(gate)
    
    println("Output norm after gate: $(norm(output))")
end

test_attention_math()
