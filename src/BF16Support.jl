# --- BF16 Support for Native Inference ---
# This module provides BFloat16 type support for efficient CPU inference
# BFloat16 uses 16 bits with same exponent range as Float32 but reduced mantissa

module BF16Support

using BFloat16s
using LinearAlgebra

export BFloat16, to_bfloat16, from_bfloat16
export bfloat16_matmul!, bfloat16_rmsnorm!, bfloat16_silu!

"""
    BFloat16

Alias for BFloat16s.BFloat16 - a 16-bit floating point type with
- 1 sign bit
- 8 exponent bits (same as Float32)
- 7 mantissa bits (vs 23 for Float32)

This provides ~50% memory reduction with minimal accuracy loss for inference.
"""
const BFloat16 = BFloat16s.BFloat16

"""
    to_bfloat16(x::AbstractArray{Float32}) -> Array{BFloat16}

Convert Float32 array to BFloat16.
"""
function to_bfloat16(x::AbstractArray{Float32})
    return BFloat16.(x)
end

"""
    from_bfloat16(x::AbstractArray{BFloat16}) -> Array{Float32}

Convert BFloat16 array to Float32. This is done for operations that
require full precision (like accumulation in matmul).
"""
function from_bfloat16(x::AbstractArray{BFloat16})
    return Float32.(x)
end

"""
    bfloat16_matmul!(C, A, B)

Matrix multiplication with BFloat16 inputs.
Performance: On CPUs without native BF16 support, converts to F32 internally.
For CPUs with AMX/AVX-512 BF16 support, this could be optimized.
"""
function bfloat16_matmul!(C::AbstractVecOrMat{Float32}, 
                          A::AbstractMatrix{BFloat16}, 
                          B::AbstractVecOrMat{BFloat16})
    # Convert to Float32 for the actual computation
    # The key benefit is memory bandwidth: BF16 weights are half the size
    A_f32 = Float32.(A)
    B_f32 = Float32.(B)
    mul!(C, A_f32, B_f32)
    return C
end

function bfloat16_matmul!(C::AbstractVector{Float32},
                          A::AbstractMatrix{BFloat16},
                          B::AbstractVector{BFloat16})
    A_f32 = Float32.(A)
    B_f32 = Float32.(B)
    mul!(C, A_f32, B_f32)
    return C
end

"""
    bfloat16_rmsnorm!(out, x, weight, eps)

In-place RMSNorm with BFloat16 support.
Converts to Float32 for numerical stability, converts back.
"""
function bfloat16_rmsnorm!(out::AbstractVector{BFloat16},
                           x::AbstractVector{BFloat16},
                           weight::AbstractVector{Float32},
                           eps::Float32)
    x_f32 = Float32.(x)
    out_f32 = Vector{Float32}(undef, length(out))
    
    # Compute RMS
    ms = zero(Float32)
    @simd for i in 1:length(x)
        ms += x_f32[i] * x_f32[i]
    end
    rms = sqrt(ms / length(x) + eps)
    inv_rms = 1.0f0 / rms
    
    # Normalize and scale
    @simd for i in 1:length(x)
        out_f32[i] = x_f32[i] * inv_rms * (weight[i] + 1.0f0)
    end
    
    # Convert back to BFloat16
    @simd for i in 1:length(out)
        out[i] = BFloat16(out_f32[i])
    end
    
    return out
end

"""
    bfloat16_silu!(out, x)

In-place SiLU activation with BFloat16 support.
"""
function bfloat16_silu!(out::AbstractVector{BFloat16},
                        x::AbstractVector{BFloat16})
    @simd for i in 1:length(x)
        xi = Float32(x[i])
        out[i] = BFloat16(xi * 1.0f0 / (1.0f0 + exp(-xi)))
    end
    return out
end

"""
    bfloat16_softmax!(scores, scale)

In-place softmax with BFloat16 inputs.
Converts to Float32 for numerical stability of exp().
"""
function bfloat16_softmax!(scores::AbstractVector{BFloat16}, scale::Float32)
    n = length(scores)
    
    # Convert to Float32
    scores_f32 = Float32.(scores)
    
    # Find max for numerical stability
    max_score = maximum(scores_f32)
    
    # Compute exp and sum
    sum_exp = zero(Float32)
    @simd for i in 1:n
        scores_f32[i] = exp((scores_f32[i] - max_score) * scale)
        sum_exp += scores_f32[i]
    end
    
    # Normalize
    inv_sum = 1.0f0 / sum_exp
    @simd for i in 1:n
        scores[i] = BFloat16(scores_f32[i] * inv_sum)
    end
    
    return scores
end

"""
Inference precision configuration.

:bf16 - Use BFloat16 for weights and activations where possible
:f32  - Use Float32 (default)
:auto - Choose based on hardware capabilities
"""
const INFERENCE_PRECISION = Ref{Symbol}(:f32)

function set_inference_precision!(prec::Symbol)
    @assert prec in (:bf16, :f32, :auto) "Precision must be :bf16, :f32, or :auto"
    INFERENCE_PRECISION[] = prec
end

function get_inference_precision()
    return INFERENCE_PRECISION[]
end

"""
    should_use_bf16() -> Bool

Determine if BF16 should be used based on current configuration.
"""
function should_use_bf16()
    prec = INFERENCE_PRECISION[]
    if prec == :bf16
        return true
    elseif prec == :f32
        return false
    else # :auto
        # Check for AMX or AVX-512 BF16 support
        # For now, default to false for :auto
        return false
    end
end

end # module
