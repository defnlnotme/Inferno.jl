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

## Recommendation: Cannot Switch to BFloat16 (GPU Limitation)

### CRITICAL BLOCKER: oneAPI/SPIR-V Does Not Support BFloat16

**Error encountered:**
```
RequiresExtension: Feature requires the following SPIR-V extension:
 SPV_KHR_bfloat16
NOTE: LLVM module contains bfloat type, translation of which requires this extension
ERROR: Failed to translate LLVM code to SPIR-V.
```

**Technical Details:**
- Intel Arc GPUs require SPV_KHR_bfloat16 extension for BF16
- This extension is not available in current oneAPI/SPIR-V implementations
- BFloat16s.jl package works on CPU but fails on GPU compilation

**Verified:**
```julia
# CPU works fine
julia> BFloat16(1.0)
BFloat16(1.0)

# GPU fails with SPV_KHR_bfloat16 extension error
julia> oneAPI.ones(BFloat16, 10)
ERROR: Failed to translate LLVM code to SPIR-V.
```

### Current Hardware Support

| Platform | BFloat16 Native | Status |
|----------|-----------------|--------|
| NVIDIA RTX 30/40 series | ✅ Yes (Tensor Cores) | Supported in CUDA |
| Intel Arc (oneAPI) | ❌ No | Requires SPV_KHR_bfloat16 |
| AMD RDNA3+ | ✅ Yes | Supported in ROCm |

### Conclusion: Must Keep Float16

**Our current approach is correct:**
- Store as Float16 (supported on Intel Arc)
- Compute with Float32 intermediate (prevents overflow)
- This matches llama.cpp's numerical approach

**Cannot switch to BF16 because:**
1. oneAPI doesn't support SPV_KHR_bfloat16 extension
2. Intel Arc GPUs don't have native BF16 instructions
3. Would require fallback to CPU (defeats purpose of GPU acceleration)

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
