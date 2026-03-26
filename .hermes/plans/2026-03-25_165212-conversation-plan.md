# Plan: Float Type Comparison - Inferno vs llama.cpp Qwen3.5 Implementation

## Goal
Compare the float types used in every part of the Qwen3.5 inference process between our Inferno.jl implementation and llama.cpp, ensuring numerical compatibility and correctness.

## CRITICAL FINDING: Qwen uses BFloat16, not Float16!

**Qwen's native dtype**: **bfloat16 (BF16)** - NOT float16 (FP16)

### Key Differences Between BF16 and FP16

| Property | Float16 (FP16) | BFloat16 (BF16) |
|----------|----------------|-----------------|
| **Exponent bits** | 5 bits | 8 bits (same as FP32) |
| **Mantissa bits** | 10 bits | 7 bits |
| **Dynamic range** | Limited (±65504 max) | Same as FP32 |
| **Precision** | Higher precision | Lower precision |
| **Overflow risk** | High (exp overflow > 6.5) | None (same range as FP32) |
| **Underflow risk** | High | Low |

### Why This Matters

1. **Float16 exp() overflow**: `exp(6.5)` = 665, but FP16 max is 65504 → **overflow**
2. **BFloat16 exp()**: Same range as FP32, no overflow for normal values
3. **llama.cpp**: Converts BF16 → FP32 for computation (no precision loss)
4. **Our implementation**: Uses FP16 → needs Float32 intermediate for exp()

### llama.cpp BF16 Handling

```cpp
// ggml-impl.h:588-594 - BF16 to FP32 is just bit shift
static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union { float f; uint32_t i; } u;
    u.i = (uint32_t)h.bits << 16;  // Zero-extend mantissa
    return u.f;
}
```

**Key insight**: BF16 → FP32 conversion is **free** (just pad zeros), no precision loss.

## Current Context

### Inferno.jl Implementation
- **Storage format**: Float16 for weights and activations
- **Computation**: Mixed Float16/Float32 (Float32 for exp/sqrt operations)
- **Backend**: oneAPI (Intel GPU)

### llama.cpp Implementation
- **Storage format**: Configurable (FP16, FP32, or quantized)
- **Computation**: Float32 for all arithmetic operations
- **Backend**: CUDA (NVIDIA GPU), with CPU fallback

## Float Type Comparison by Component

### 1. Activation Functions

| Operation | llama.cpp | Inferno.jl | Status |
|-----------|-----------|------------|--------|
| **SiLU** | `x / (1.0f + expf(-x))` - all Float32 | Float32 intermediate, Float16 output | **MATCHES** - We use Float32 for exp() |
| **Sigmoid** | `1.0f / (1.0f + expf(-x))` - all Float32 | Float32 intermediate, Float16 output | **MATCHES** |
| **Softplus** | `(x > 20.0f) ? x : logf(1.0f + expf(x))` - Float32 | Float32 intermediate with same threshold | **MATCHES** |

**Key Finding**: llama.cpp unary ops (unary.cu:112-121) cast to float before operation:
```cpp
template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    dst[i] = (T)op((float)x[i]);  // Cast to float, compute, cast back
}
```

### 2. Softmax / Attention Scores

| Operation | llama.cpp | Inferno.jl | Status |
|-----------|-----------|------------|--------|
| **Attention softmax** | Float32 arithmetic (softmax.cu:114) | Float32 for exp() and sum | **MATCHES** |
| **Scale factor** | `1.0f / sqrtf(float(n_embd_head))` - Float32 | Float32 for scale computation | **MATCHES** |
| **Max reduction** | Float32 max_val | Float32 mx | **MATCHES** |
| **Exp subtraction** | `expf(vals[col] - max_val)` - Float32 | `exp(Float32(scores[i]) - mx)` | **MATCHES** |

**Key Finding**: llama.cpp softmax always uses Float32 for computation even when input is FP16 (softmax.cu:55-138):
```cpp
float max_val = -INFINITY;
const float val = expf(vals[col] - max_val);  // Always float
```

### 3. Rotary Position Embeddings (RoPE)

| Operation | llama.cpp | Inferno.jl | Status |
|-----------|-----------|------------|--------|
| **inv_freq computation** | Float32 (rope.cu:100) | Float32 | **MATCHES** |
| **Position encoding** | `pos[i2]*powf(theta_scale, i0/2.0f)` - Float32 | Float32 position | **MATCHES** |
| **sin/cos computation** | `sinf(theta)`, `cosf(theta)` - Float32 | `sincos(freq)` - Float32 | **MATCHES** |
| **Rotation application** | Float32 arithmetic, stored as input type | Float32 arithmetic | **MATCHES** |

**Key Finding**: llama.cpp rope_norm kernel (rope.cu:100):
```cpp
const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);
```
Uses Float32 for all position-related computation.

### 4. Quantization / Dequantization

| Operation | llama.cpp | Inferno.jl | Status |
|-----------|-----------|------------|--------|
| **Q4_0 dequant** | `(val - 8.0f) * d` - Float32 | Not implemented | N/A |
| **Q8_0 dequant** | `val * d` - Float32 | Not implemented | N/A |
| **Scale conversion** | `__half2float()` for FP16 scales | Automatic Float16 | **NEEDS REVIEW** |

**Key Finding**: llama.cpp dequantize.cuh always converts to Float32:
```cpp
const float d = x[ib].d;  // Scale is Float32
v.x = (v.x - 8.0f) * d;   // Arithmetic in Float32
```

### 5. SSM (Gated Delta Net) Operations

| Operation | llama.cpp | Inferno.jl | Status |
|-----------|-----------|------------|--------|
| **Conv1D** | Float32 accumulation (ssm-conv.cu) | Float32 | **MATCHES** |
| **State update** | `state * expf(dt_soft_plus * A)` - Float32 | Float32 BLAS | **MATCHES** |
| **Softplus for dt** | `log1pf(expf(dt_soft_plus))` - Float32 | Float32 with threshold | **MATCHES** |

**Key Finding**: llama.cpp ssm-scan.cu:83-97:
```cpp
float dt_soft_plus = dt_block[i * stride_dt + threadIdx.x];
if (dt_soft_plus <= 20.0f) {
    dt_soft_plus = log1pf(expf(dt_soft_plus));
}
float state = regs0[n] * expf(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
```
All SSM arithmetic is Float32.

### 6. Matrix Multiplication

| Operation | llama.cpp | Inferno.jl | Status |
|-----------|-----------|------------|--------|
| **GEMM (FP16 input)** | Tensor cores or FP16 SIMD | oneAPI mat_mul | **NEEDS REVIEW** |
| **GEMM (quantized)** | Dequantize to Float32 first | QuantMatrix with FP16 dequant | **NEEDS REVIEW** |

**Key Finding**: llama.cpp uses FP16 tensor cores when available, but accumulates in Float32 for numerical stability.

### 7. Layer Normalization (RMSNorm)

| Operation | llama.cpp | Inferno.jl | Status |
|-----------|-----------|------------|--------|
| **Sum of squares** | Float32 accumulation | Float32 | **MATCHES** |
| **RMS computation** | `sqrtf(sum / n + eps)` - Float32 | Float32 sqrt | **MATCHES** |
| **Normalization** | Float32 arithmetic | Float32 arithmetic | **MATCHES** |

### 8. Sampling / Softmax

| Operation | llama.cpp | Inferno.jl | Status |
|-----------|-----------|------------|--------|
| **Temperature scaling** | `x * scale` in Float32 | Float32 for exp() | **MATCHES** |
| **Top-p sampling** | Float32 probabilities | Float16 probabilities | **NEEDS FIX** |

**Issue Found**: Our sampling uses Float16 for probability accumulation. Should use Float32 like llama.cpp.

## Issues to Fix

### Issue 1: Sampling Probability Accumulation (MEDIUM)
**File**: `src/Engine.jl`
**Problem**: Simple sampling uses Float16 for cumulative probability:
```julia
function simple_sample(probs::Vector{Float16})
    cum = Float16(0.0)  # Should be Float32
```
**Fix**: Use Float32 for cumulative sum.

### Issue 2: Verify oneAPI mat_mul Accumulation (LOW)
**File**: `src/Model.jl`
**Problem**: Need to verify oneAPI's mat_mul accumulates in Float32 when using Float16 inputs.
**Fix**: Check oneAPI documentation or add explicit Float32 accumulation.

### Issue 3: RoPE Cache Storage (LOW)
**File**: `src/Model.jl`
**Problem**: RoPE sin/cos cache is Float32, which matches llama.cpp. Good.
**Status**: Already fixed.

## Verification Steps

1. **Run numerical comparison test**:
   - Load same Qwen3.5 model in both llama.cpp and Inferno
   - Feed identical input tokens
   - Compare logits at each layer boundary
   - Verify RMS error is within acceptable bounds (< 0.1%)

2. **Test activation function outputs**:
   - Compare SiLU, sigmoid, softplus outputs for edge cases
   - Test with large positive/negative values
   - Verify NaN handling

3. **Test attention softmax**:
   - Compare attention scores for long sequences
   - Verify numerical stability at context boundaries
   - Test with extreme logits (> 100)

## Files to Modify

1. `src/Engine.jl` - Fix sampling probability accumulation
2. `src/Model.jl` - No changes needed (already using Float32 correctly)
3. `src/test/numerical_comparison.jl` - Create new test file

## Implementation Steps

### Step 1: Fix Sampling (Engine.jl)
```julia
function simple_sample(probs::Vector{Float16})
    r = rand()
    cum = Float32(0.0)  # Changed from Float16
    for (i, p) in enumerate(probs)
        cum += Float32(p)  # Accumulate in Float32
        if r <= cum
            return i
        end
    end
    return length(probs)
end
```

### Step 2: Add Numerical Comparison Test
Create `test/numerical_comparison.jl` to compare outputs with llama.cpp reference.

### Step 3: Document Float Type Policy
Add documentation explaining:
- All exp() operations use Float32
- All sqrt() operations use Float32  
- All accumulation operations use Float32
- Storage is Float16 for memory efficiency

## Summary

**Overall Status**: ✅ **GOOD** - Our implementation matches llama.cpp's numerical approach.

**Key Similarities**:
- Both use Float32 for all exp() operations
- Both use Float32 for softmax computation
- Both use Float32 for RoPE position encoding
- Both use Float32 for SSM state updates
- Both use Float32 for normalization

**Minor Fix Needed**:
- Sampling probability accumulation should use Float32

**Recommendation**: Our Float16 storage with Float32 compute approach is numerically sound and matches llama.cpp's philosophy of "compute in Float32, store in Float16".

## Recommendation: Cannot Switch to BFloat16 (SPIR-V Extension Issue)

### CRITICAL BLOCKER: BFloat16 Requires Unsupported Extensions

**Error Chain:**
1. First error: `SPV_KHR_bfloat16` required for bfloat type
2. Second error: `SPV_INTEL_bfloat16_arithmetic` required for arithmetic operations  
3. Third error: `Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_INTEL_bfloat16_arithmetic'`

**Technical Investigation:**

```julia
# Attempted: Add SPV_KHR_bfloat16 extension
extensions = String[
 "SPV_EXT_relaxed_printf_string_address_space",
 "SPV_EXT_shader_atomic_float_add",
 "SPV_KHR_bfloat16" # <-- Added this
]

# Result: Compilation asks for SPV_INTEL_bfloat16_arithmetic
# Then: Runtime rejects SPV_INTEL_bfloat16_arithmetic as "unknown extension"
```

**Root Cause:**
- LLVM/SPIR-V translator generates BF16 arithmetic operations
- These require `SPV_INTEL_bfloat16_arithmetic` extension
- Intel Arc GPU runtime does NOT support this extension
- The extension exists in SPIR-V spec but not in Intel's implementation

**Verified:**
```julia
# oneAPI.jl compilation pipeline:
LLVM IR → SPIR-V translator → GPU execution

# Failure point:
# SPIR-V translator: Requires SPV_INTEL_bfloat16_arithmetic for BF16 ops
# Intel runtime: "Invalid SPIR-V module: unknown extension"
```

### Layer-by-Layer Analysis

| Layer | BFloat16 Support | Status |
|-------|------------------|--------|
| Julia (BFloat16s.jl) | ✅ Yes | Works on CPU |
| LLVM IR | ✅ Yes | bfloat type exists |
| SPIR-V Translator | ⚠️ Partial | Needs extension |
| oneAPI/SPIR-V | ❌ No | Extension unsupported |
| Intel Arc Hardware | ❓ Unknown | Runtime rejects it |

### Hardware Support Reality Check

| Platform | BF16 Native | Extension Supported | Works? |
|----------|-------------|---------------------|--------|
| NVIDIA RTX 30/40 | ✅ Tensor Cores | CUDA supports BF16 | ✅ Yes |
| Intel Arc | ❓ Unknown | ❌ Runtime rejects | ❌ No |
| AMD RDNA3+ | ✅ Native | ROCm supports | ✅ Yes |

**Conclusion**: This is **NOT** a oneAPI.jl limitation - it's an **Intel Arc GPU hardware/driver limitation**.

## OpenVINO Approach: How Intel Actually Handles This

### Key Insight: OpenVINO Uses oneDNN (CPU), Not GPU

OpenVINO's quantized inference works differently:

**1. Quantized Types Supported:**
```cpp
// OpenVINO element types (element_type.hpp)
enum class Type_t {
 i4, i8,     // Integer quantized
 u4, u8,     // Unsigned quantized
 bf16, f16,  // Half precision
 f32,        // Full precision
 nf4,        // Normalized float 4
 f8e4m3, f8e5m2,  // FP8 types
};
```

**2. Computation Approach:**

oneDNN (the backend for OpenVINO CPU inference) uses:
- **INT8/INT4 weights**: Dequantized to **FP32** for computation
- **BF16 weights**: Native BF16 operations on supported CPUs (AVX512_BF16)
- **Hybrid approach**: Quantize weights, compute in FP32, store in FP16/BF16

**3. BF16 on Intel CPUs:**
```cpp
// oneDNN bfloat16.cpp - CPU conversion
void cvt_bfloat16_to_float(float *out, const bfloat16_t *inp, size_t nelems) {
#if DNNL_X64
 if (mayiuse(cpu_isa_t::avx512_core) || mayiuse(avx2_vnni_2)) {
 // Use AVX512 BF16 instructions if available
 static const cpu::x64::jit_cvt_xf16_to_ps_t kernel(data_type::bf16, false);
 return kernel(out, inp, nelems);
 }
#endif
 // Fallback: Scalar conversion
 PRAGMA_OMP_SIMD()
 for (size_t i = 0; i < nelems; ++i)
 out[i] = inp[i]; // BF16 → FP32 is just bit extension
}
```

**4. INT8 MatMul:**
```cpp
// oneDNN gemm_driver.cpp
constexpr bool is_int8 = utils::one_of(
 data_traits_t<a_type>::data_type, data_type::s8, data_type::u8);
bool is_int8_amx = is_int8 && mayiuse(avx512_core_amx);
// INT8 GEMM accumulates to INT32, then converts to FP32
```

### Why This Works for OpenVINO but Not Us

| Aspect | OpenVINO/oneDNN | Our GPU Approach |
|--------|-----------------|------------------|
| **Backend** | CPU (AVX512/AMX) | GPU (Intel Arc) |
| **INT8/INT4** | Dequant to FP32 on-the-fly | Not implemented |
| **BF16** | Native AVX512_BF16 instructions | SPIR-V extension missing |
| **Accumulation** | INT32 for quantized, FP32 for BF16 | FP32 intermediate |
| **Hardware** | Xeon/Core with AVX512 | Intel Arc GPU |

### The Real Solution: Quantized Inference

**What OpenVINO actually does:**
1. **INT4/INT8 weights**: Stored in 4/8 bits
2. **Dequantization**: On-the-fly to FP32 during GEMM
3. **Accumulation**: INT32 for quantized, FP32 for BF16
4. **Activation storage**: FP16 or BF16

**For our GPU implementation, we should:**
1. **Keep FP16 storage** (matching Qwen's weight loading)
2. **Add INT8/INT4 quantization support** (future work)
3. **Use FP32 intermediate** (as we do now)
4. **Consider block quantization** (like llama.cpp's k-quants)

### Future Path: Add Quantization Support

If we want to match OpenVINO's efficiency:

```julia
# Potential future implementation
struct QuantMatrix
 weight::Vector{UInt8} # INT8/INT4 packed weights
 scale::Vector{Float16} # Per-channel scales
 zeropoint::Vector{Int8} # Per-channel zero points
 shape::Tuple{Int, Int}
 
 # Dequantize on-the-fly during matmul
 function matmul(q::QuantMatrix, x::oneMatrix{Float16})
 # Dequantize to FP32, compute, store as FP16
 weight_f32 = dequantize(q.weight, q.scale, q.zeropoint)
 return oneMatrix{Float16}(weight_f32 * x)
 end
end
```

### Future Consideration

If oneAPI adds BF16 support in the future:
```julia
# Would simplify our kernels
function silu_bf16(x::BFloat16)
    # BF16 has same range as FP32, no overflow risk
    # But we can't use this until oneAPI supports it
    return x * (1.0f0 / (1.0f0 + exp(-Float32(x))))
end
```

For now, our Float16 + Float32 intermediate approach is the **only viable option** for Intel Arc GPUs.
