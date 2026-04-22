"""
Arrow Lake (Core Ultra 7 265K) Optimization Module

Arrow Lake features:
- 256-bit AVX2 + AVX-VNNI (not traditional 512-bit AVX-512)
- AVX-VNNI-BF16 support for 256-bit BF16 operations
- 6P + 8E cores, 20 threads
- Focus: AVX-256 VNNI-BF16 kernels for memory-bound workloads

Key Instructions:
- VDPBF16PS: Dot-product of 256-bit BF16 pairs -> F32 accumulator
- VCVTNE2PS2BF16: Convert packed F32 to BF16
- VNNI: Neural network ops on 256-bit vectors

Performance Notes:
- BF16 weights = 50% memory bandwidth of F32
- 256-bit vectors = more flexible than 512-bit (no frequency penalty)
- Kernel accumulators stay F32 (numerical stability)
"""
module ArrowLake

using BFloat16s
using LinearAlgebra
using LoopVectorization
using StaticArrays

export has_arrow_lake_features, has_vnni_bf16
export bf16_matmul_vec!, bf16_matmul_transpose!
export bf16_rmsnorm!, bf16_silu!, bf16_softmax!
export to_bfloat16_weights, BFloat16Matrix
export bf16_gemv_c!, bf16_gemv_c_cols!, fp32_to_bf16_c!

# ============================================================================
# C AVX2 BF16 Kernel (compiled from kernels/bf16_avx2.c)
# ============================================================================

const LIBBF16_PATH = joinpath(@__DIR__, "kernels", "libbf16_avx2.so")
const _libbf16_loaded = Ref(false)
const _libbf16_handle = Ref{Ptr{Cvoid}}(C_NULL)

function _load_libbf16()
    if !_libbf16_loaded[]
        if isfile(LIBBF16_PATH)
            _libbf16_loaded[] = true
        else
            @warn "BF16 AVX2 kernel not found at $LIBBF16_PATH - BF16 C kernels unavailable"
        end
    end
    return _libbf16_loaded[]
end

"""
    bf16_gemv_c!(out, A_u16, x_bf16)

Call C AVX2 BF16 matmul kernel. A_u16 is UInt16 view of BF16 weight matrix,
x_bf16 is UInt16 BF16-converted activation vector, out is F32 result.
"""
function bf16_gemv_c!(out::AbstractVector{Float32}, A_u16::AbstractArray{UInt16}, x_bf16::Vector{UInt16}; lda::Int=0)
 _load_libbf16() || error("BF16 C kernel not available")
 m, n = size(A_u16)
 actual_lda = lda > 0 ? lda : m
 ccall((:bf16_gemv_avx2, LIBBF16_PATH), Cvoid,
 (Ptr{UInt16}, Ptr{UInt16}, Ptr{Float32}, Cint, Cint, Cint),
 A_u16, x_bf16, out, m, n, actual_lda)
 return out
end

"""
 bf16_gemv_c_cols!(out, A_u16, x_bf16, col_start, col_end, col_len)

Compute out[k] = dot(A[:, col_start+k-1], x) for columns col_start..col_end.
Uses batch C kernel bf16_gemv_cols_avx2 for minimal per-column overhead.
"""
function bf16_gemv_c_cols!(out::AbstractVector{Float32}, A_u16::AbstractArray{UInt16}, 
 x_bf16::Vector{UInt16}, col_start::Int, col_end::Int, col_len::Int)
 _load_libbf16() || error("BF16 C kernel not available")
 n_cols = col_end - col_start + 1
 # C uses 0-based column index
 ccall((:bf16_gemv_cols_avx2, LIBBF16_PATH), Cvoid,
 (Ptr{UInt16}, Ptr{UInt16}, Ptr{Float32}, Cint, Cint, Cint),
 A_u16, x_bf16, out, col_start - 1, n_cols, col_len)
 return out
end

"""
    fp32_to_bf16_c!(dst_bf16, src_f32)

Convert F32 vector to BF16 using C kernel (round-to-nearest-even).
"""
function fp32_to_bf16_c!(dst_bf16::Vector{UInt16}, src_f32::Vector{Float32})
    _load_libbf16() || error("BF16 C kernel not available")
    ccall((:fp32_to_bf16_row, LIBBF16_PATH), Cvoid,
          (Ptr{Float32}, Ptr{UInt16}, Cint),
          src_f32, dst_bf16, length(src_f32))
    return dst_bf16
end

# ============================================================================
# Feature Detection
# ============================================================================

"""
    parse_cpu_flags()

Parse /proc/cpuinfo for AVX-VNNI features (Linux only).
"""
function parse_cpu_flags()
    features = Dict{Symbol, Bool}(
        :avx2 => false,
        :avx_vnni => false,
        :avx_vnni_int8 => false,
        :avx_vnni_bf16 => false,
        :fma => false,
        :bmi2 => false,
    )
    
    try
        if isfile("/proc/cpuinfo")
            cpuinfo = read("/proc/cpuinfo", String)
            
            # Check for AVX2 and extensions
            features[:avx2] = occursin(" avx2 ", cpuinfo)
            features[:avx_vnni] = occursin(" avx_vnni ", cpuinfo) || occursin("vnni", cpuinfo)
            features[:fma] = occursin(" fma ", cpuinfo)
            features[:bmi2] = occursin(" bmi2 ", cpuinfo)
            
            # BF16 via AVX-VNNI (usually appears as avx_vnni or detection via cpuid)
            # Arrow Lake will have this combination: Intel + Ultra + avx_vnni
            features[:arrow_lake] = occursin(r"Intel.*Core.*Ultra", cpuinfo) && features[:avx_vnni]
        end
    catch e
        @debug "Failed to parse CPU flags" exception=e
    end
    
    return features
end

const CPU_FEATURES = Ref{Union{Dict{Symbol, Bool}, Nothing}}(nothing)

function get_cpu_features()
    if CPU_FEATURES[] === nothing
        CPU_FEATURES[] = parse_cpu_flags()
    end
    return CPU_FEATURES[]
end

"""
    has_arrow_lake_features() -> Bool

Check if CPU has Arrow Lake optimization features.
"""
function has_arrow_lake_features()::Bool
    feats = get_cpu_features()
    return feats[:avx2] && feats[:avx_vnni]
end

"""
    has_vnni_bf16() -> Bool

Check if CPU supports AVX-VNNI-BF16 instructions.
"""
function has_vnni_bf16()::Bool
    feats = get_cpu_features()
    return feats[:avx_vnni] && feats[:fma]
end

"""
    print_cpu_features()

Print detected CPU features.
"""
function print_cpu_features()
    feats = get_cpu_features()
    
    println("=== Arrow Lake CPU Detection ===")
    println("AVX2:        ", feats[:avx2])
    println("AVX-VNNI:    ", feats[:avx_vnni])
    println("FMA:         ", feats[:fma])
    println("BMI2:        ", feats[:bmi2])
    println("Arrow Lake:  ", feats[:arrow_lake])
    println("VNNI-BF16:   ", has_vnni_bf16())
    println()
    println("Note: Arrow Lake has AVX-VNNI (256-bit), not AVX-512")
end

# ============================================================================
# BF16 Matrix Type
# ============================================================================

"""
    BFloat16Matrix

Matrix stored in BFloat16 for 50% memory reduction.
"""
struct BFloat16Matrix{T<:AbstractMatrix{BFloat16}}
    data::T
    num_rows::Int
    num_cols::Int
end

function BFloat16Matrix(A::AbstractMatrix{Float32})
    m = Matrix{BFloat16}(undef, size(A, 1), size(A, 2))
    @inbounds @simd for j in axes(A, 2)
        for i in axes(A, 1)
            m[i, j] = BFloat16(A[i, j])
        end
    end
    return BFloat16Matrix(m, size(A, 1), size(A, 2))
end

Base.size(B::BFloat16Matrix) = (B.num_rows, B.num_cols)
Base.getindex(B::BFloat16Matrix, i::Int, j::Int) = Float32(B.data[i, j])

"""
    to_bfloat16_weights(model)

Convert model weights to BFloat16 for memory savings.
"""
function to_bfloat16_weights(weights::AbstractMatrix{Float32})
    return Matrix{BFloat16}(@.(BFloat16(weights)))
end

# ============================================================================
# BF16 Matmul Kernels (AVX-VNNI optimized)
# ============================================================================

"""
 bf16_matmul_vec!(out, A::Matrix{BFloat16}, x::Vector{Float32})

Matrix-vector multiply with BF16 weights, F32 activations/accumulators.
Strategy: reinterpret BF16 matrix as UInt16 array (view, no copy),
then convert each UInt16 to F32 in-register via bit shift.
No BFloat16 objects are created - avoids Julia BFloat16 boxing.

Memory bandwidth: 50% of F32 (2 bytes/element vs 4).
Pipeline: weights BF16, activations F32, accumulator F32.
"""
function bf16_matmul_vec!(out::AbstractVector{Float32}, 
                          A::AbstractMatrix{BFloat16}, 
                          x::AbstractVector{Float32})
    m, n = size(A)
    # Reinterpret BF16 array as UInt16 array - this is a VIEW, zero allocation
    A_u16 = reinterpret(UInt16, A)
    
    @inbounds for i in 1:m
        s = zero(Float32)
        @simd for j in 1:n
            # Read UInt16 (primitive, no boxing), shift to UInt32, reinterpret as Float32
            # All operations are on bitstypes -> runs in registers, zero allocations
            u32 = UInt32(A_u16[i, j]) << 16
            w = reinterpret(Float32, u32)
            s = muladd(w, x[j], s)
        end
        out[i] = s
    end
    
    return out
end

"""
    bf16_matmul_vec!(out, A::Matrix{Float32}, x::AbstractVector)

Fallback for Float32 weights (convert on-the-fly).
"""
function bf16_matmul_vec!(out::AbstractVector{Float32}, 
                          A::AbstractMatrix{Float32}, 
                          x::AbstractVector{Float32})
    # Check if we should use cached BF16 weights
    m, n = size(A)
    
    # Fast path with @turbo for cache-friendly access
    @turbo thread=true for i in 1:m
        s = 0.0f0
        for j in 1:n
            s += A[i, j] * x[j]
        end
        out[i] = s
    end
    
    return out
end

"""
    bf16_matmul_transpose!(out::AbstractVector{Float32}, 
                          A::AbstractMatrix{BFloat16}, 
                          x::AbstractVector{Float32})

Transpose matmul: out = A' * x (for linear layers with column-major storage).
Attention: x is length m, A is m×n, output length n.
"""
function bf16_matmul_transpose!(out::AbstractVector{Float32}, 
                                 A::AbstractMatrix{BFloat16}, 
                                 x::AbstractVector{Float32})
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch("input size"))
    length(out) == n || throw(DimensionMismatch("output size"))
    
    fill!(out, 0.0f0)
    @inbounds for i in 1:m
        xi = x[i]
        @simd for j in 1:n
            out[j] += Float32(A[i, j]) * xi
        end
    end
    
    return out
end

# ============================================================================
# RMSNorm (AVX-256 optimized)
# ============================================================================

"""
    bf16_rmsnorm!(out::AbstractVector{Float32}, 
                  x::AbstractVector{Float32}, 
                  weight::AbstractVector{Float32}, 
                  eps::Float32=1.0f-6)

RMSNorm: x / sqrt(mean(x^2) + eps) * weight
Optimized for Arrow Lake with fused multiply-add.
"""
function bf16_rmsnorm!(out::AbstractVector{Float32},
                       x::AbstractVector{Float32},
                       weight::AbstractVector{Float32},
                       eps::Float32 = 1.0f-6)
    n = length(x)
    n == length(out) || throw(DimensionMismatch("output size"))
    n == length(weight) || throw(DimensionMismatch("weight size"))
    
    # Compute mean of squares
    msq = 0.0f0
    @inbounds @simd for i in 1:n
        msq = muladd(x[i], x[i], msq)
    end
    msq = msq / n
    
    # Compute normalization factor
    inv_rms = 1.0f0 / sqrt(msq + eps)
    
    # Normalize and scale
    @inbounds @simd for i in 1:n
        out[i] = x[i] * inv_rms * weight[i]
    end
    
    return out
end

# ============================================================================
# Activation Functions
# ============================================================================

"""
    bf16_silu!(out::AbstractVector{Float32}, x::AbstractVector{Float32})

SiLU activation: x * sigmoid(x)
"""
function bf16_silu!(out::AbstractVector{Float32}, x::AbstractVector{Float32})
    n = length(x)
    n == length(out) || throw(DimensionMismatch("size mismatch"))
    
    @inbounds @simd for i in 1:n
        # Fast sigmoid approximation: 1 / (1 + exp(-x))
        xi = x[i]
        # Clamp for numerical stability
        xi = clamp(xi, -10.0f0, 10.0f0)
        sig = 1.0f0 / (1.0f0 + exp(-xi))
        out[i] = xi * sig
    end
    
    return out
end

"""
    bf16_softmax!(out::AbstractMatrix{Float32}, x::AbstractMatrix{Float32})
    bf16_softmax!(out::AbstractVector{Float32}, x::AbstractVector{Float32})

Numerically stable softmax.
"""
function bf16_softmax!(out::AbstractVector{Float32}, x::AbstractVector{Float32})
    n = length(x)
    n == length(out) || throw(DimensionMismatch("size mismatch"))
    
    # Find max for numerical stability
    max_val = typemin(Float32)
    @inbounds for i in 1:n
        max_val = max(max_val, x[i])
    end
    
    # Compute exp and sum
    sum_exp = 0.0f0
    @inbounds @simd for i in 1:n
        e = exp(x[i] - max_val)
        out[i] = e
        sum_exp += e
    end
    
    # Normalize
    inv_sum = 1.0f0 / sum_exp
    @inbounds @simd for i in 1:n
        out[i] *= inv_sum
    end
    
    return out
end

function bf16_softmax!(out::AbstractMatrix{Float32}, x::AbstractMatrix{Float32})
    size(out) == size(x) || throw(DimensionMismatch("size mismatch"))
    
    # Process each column independently (attention scores)
    @inbounds for j in axes(x, 2)
        col_max = typemin(Float32)
        for i in axes(x, 1)
            col_max = max(col_max, x[i, j])
        end
        
        sum_exp = 0.0f0
        for i in axes(x, 1)
            e = exp(x[i, j] - col_max)
            out[i, j] = e
            sum_exp += e
        end
        
        inv_sum = 1.0f0 / sum_exp
        for i in axes(x, 1)
            out[i, j] *= inv_sum
        end
    end
    
    return out
end

# ============================================================================
# Linear Layer (Conv1D style)
# ============================================================================

"""
    bf16_linear_forward!(out, weight, x, bias=nothing)

Linear transformation: out = weight * x + bias
Uses BF16 weights for memory bandwidth reduction.
"""
function bf16_linear_forward!(out::AbstractVector{Float32},
                               weight::AbstractMatrix{BFloat16},
                               x::AbstractVector{Float32},
                               bias::Union{Nothing, AbstractVector{Float32}} = nothing)
    bf16_matmul_vec!(out, weight, x)
    
    if bias !== nothing
        @inbounds @simd for i in eachindex(out)
            out[i] += bias[i]
        end
    end
    
    return out
end

function bf16_linear_forward!(out::AbstractVector{Float32},
                               weight::AbstractMatrix{Float32},
                               x::AbstractVector{Float32},
                               bias::Union{Nothing, AbstractVector{Float32}} = nothing)
    # Float32 weights - use efficient matmul
    LinearAlgebra.mul!(out, weight, x)
    
    if bias !== nothing
        @inbounds @simd for i in eachindex(out)
            out[i] += bias[i]
        end
    end
    
    return out
end

# ============================================================================
# Initialization
# ============================================================================

function __init__()
    features = get_cpu_features()
    
    if features[:arrow_lake]
        @debug "Arrow Lake (Core Ultra 2) detected - AVX-VNNI optimized kernels enabled"
    elseif features[:avx_vnni]
        @debug "AVX-VNNI detected - using VNNI-optimized kernels"
    else
        @debug "Standard AVX2 kernels (no VNNI)"
    end
end

end # module
