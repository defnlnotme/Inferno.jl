# GPU Implementation Fixes for Inferno.jl

**Date**: 2026-03-29  
**Status**: Critical issues identified requiring immediate attention

---

## Executive Summary

The GPU implementation of Qwen 3.5 inference in Inferno.jl has several critical issues that prevent correct operation. The CPU backend works but produces degraded output (mostly whitespace), indicating potential model loading or tokenizer issues. The GPU backend fails with oneAPI driver errors during model loading.

---

## Critical Issues

### 1. GPU Memory Allocation Failures (CRITICAL)

**Symptom**: `ZeError: unknown or internal error (code 2147483646, ZE_RESULT_ERROR_UNKNOWN)`

**Location**: `src/Loader.jl:90` in `get_weight()`

**Root Cause**: 
- Large tensor allocations failing on Intel GPU
- The `oneArray(Float16.(tensor'))` pattern creates temporary CPU arrays before GPU transfer
- For large models (320+ tensors), this causes memory pressure

**Fix Required**:
```julia
# Current (problematic):
function get_weight(file::GGUF.GGUFFile, name::String)
    tensor = extract_tensor(file, name)
    if ndims(tensor) == 2
        return oneArray(Float16.(tensor'))  # Creates large temp array
    end
end

# Proposed fix - stream directly to GPU:
function get_weight(file::GGUF.GGUFFile, info::GGUF.TensorInfo)
    num_elements = Int(prod(info.dimensions))
    start = Int(file.data_offset + info.offset) + 1
    
    # For large tensors, allocate GPU memory first, then copy in chunks
    dims = Tuple(Int.(info.dimensions))
    
    if info.type == GGUF.GGML_TYPE_F16
        # Allocate GPU array directly
        gpu_array = oneArray{Float16}(undef, dims)
        # Copy in chunks to avoid memory pressure
        chunk_size = 1024 * 1024  # 1M elements per chunk
        for i in 1:chunk_size:num_elements
            chunk_end = min(i + chunk_size - 1, num_elements)
            chunk_len = chunk_end - i + 1
            cpu_chunk = reinterpret(Float16, @view file.tensor_data[start+i-1:start+chunk_end-1])
            gpu_array_flat = reshape(gpu_array, :)
            gpu_array_flat[i:chunk_end] .= cpu_chunk
        end
        return ndims(gpu_array) == 2 ? oneArray(Float16.(gpu_array')) : gpu_array
    end
    # ... handle other types
end
```

**Priority**: P0 - Blocks GPU usage entirely

---

### 2. SSM Layer GPU/CPU Hybrid Processing (HIGH)

**Location**: `src/Model.jl:1188-1317` in `GatedDeltaNet` call operator

**Issue**: The SSM layer copies data back and forth between GPU and CPU for every token:

```julia
# Line 1200: GPU -> CPU copy
qkv_cpu = vec(Float32.(Array(m.qkv_proj)))

# Lines 1201-1230: All computation on CPU

# Line 1308: CPU -> GPU copy  
copyto!(m.branch_out, reshape(Float16.(m.y_all_cpu), :, 1))

# Line 1310: GPU matmul
result = mat_mul(m.ssm_out, m.branch_out)
```

**Impact**: 
- Massive PCIe bandwidth overhead
- Defeats purpose of GPU acceleration
- Likely slower than pure CPU implementation

**Fix Required**: Implement pure GPU kernels for SSM operations:

1. **Convolution kernel**: Replace CPU loop with GPU kernel
2. **State update kernel**: Fuse state decay and update into single kernel
3. **Normalization kernel**: Keep RMSNorm on GPU
4. **SiLU gating kernel**: Already exists, use consistently

**Estimated effort**: 2-3 days for complete GPU implementation

**Priority**: P1 - Major performance issue

---

### 3. FullAttention GPU Implementation Issues (HIGH)

**Location**: `src/Model.jl:656-760`

**Issues Identified**:

#### 3.1. Inconsistent Array Handling
```julia
# Line 665-667: Mixed GPU/CPU computation for gating
gate_raw_cpu = Array(gate_raw)  # GPU -> CPU
@. gate_raw_cpu = gate_raw_cpu * (Float32(1.0) / (Float32(1.0) + exp(-Float32(gate_raw_cpu))))
gate_silu = oneArray(Float16.(gate_raw_cpu))  # CPU -> GPU
q_gated = q_rope .* gate_silu
```

**Fix**: Use existing `silu_kernel!` or implement inline GPU computation:
```julia
# Use GPU kernel for SiLU gating
silu_kernel!(gate_raw, gate_raw, length(gate_raw))
q_gated = q_rope .* gate_raw
```

#### 3.2. Prefill Path Uses CPU Arrays
```julia
# Line 740: Allocates CPU array
combined_all = zeros(T, hd * m.n_heads, seq)

# Lines 742-770: Nested loops on CPU
for h in 1:m.n_heads
    for s in 1:seq
        # ... CPU computation
    end
end

# Line 778: Copy to GPU
combined = oneArray(combined_all)
```

**Fix**: Implement batched GPU attention kernel for prefill path.

**Priority**: P1 - Performance and correctness issue

---

### 4. RoPE Implementation Issues (MEDIUM)

**Location**: `src/Model.jl:253-291` in `rope_kernel!`

**Issues**:

#### 4.1. Unnecessary GPU->CPU->GPU Transfers
```julia
function rope_kernel!(x, inv_freq, pos, d, h, seq, d_rope)
    # Line 261-262: Copy to CPU
    x_cpu = Array(x)
    inv_freq_cpu = Array(inv_freq)
    
    # Lines 264-283: Compute on CPU
    
    # Line 285: Copy back to GPU
    copyto!(x, x_cpu)
end
```

**Impact**: For every attention/SSM layer, RoPE causes 2 PCIe transfers per token.

**Fix**: Implement pure GPU RoPE kernel:
```julia
function rope_kernel_gpu!(x_gpu, inv_freq_gpu, pos, d, h, seq, d_rope)
    # Launch GPU kernel that computes sin/cos and applies RoPE in-place
    # No CPU involvement
    @cuda threads=256 blocks=ceil(div(d * h * seq, 256)) rope_kernel!(
        x_gpu, inv_freq_gpu, pos, d, h, seq, d_rope
    )
end
```

#### 4.2. Missing RoPE for MLA
The MLA implementation in `MLAttention` applies RoPE correctly but the cache storage doesn't account for the split structure.

**Priority**: P2 - Performance issue

---

### 5. KV Cache Structure Mismatch for MLA (HIGH)

**Location**: `src/Model.jl:866-1043` in `MLAttention`

**Issue**: The MLA implementation stores K/V in a non-standard format:
```julia
# Line 898-900: Non-standard cache layout
@views cache.k[:, 1, pos + 1] .= k_combined  # k_combined = vcat(k_nope, k_pe)
@views cache.v[:, 1, pos + 1] .= v_combined
```

**Problem**: 
- Standard cache: `(head_dim, n_kv_heads, max_seq)`
- MLA cache: `(k_nope_dim + k_pe_dim, 1, max_seq)` - only uses first "head"
- This breaks when `n_heads > 1` (Qwen 3.5 has 8 heads)

**Fix Required**: Either:
1. **Option A**: Extend `KVCache` struct with MLA-specific fields
2. **Option B**: Use separate cache structure for MLA models

```julia
# Option A: Extend KVCache
mutable struct KVCache{T<:AbstractArray}
    k::T  # Standard: (head_dim, n_kv, max_seq)
    v::T
    # MLA-specific (optional, only allocated for MLA models)
    mla_k_nope::Union{T,Nothing}  # (qk_nope_head_dim, n_heads, max_seq)
    mla_k_pe::Union{T,Nothing}    # (qk_rope_head_dim, n_heads, max_seq)
    mla_v::Union{T,Nothing}       # (v_head_dim, n_heads, max_seq)
    pos::Int
    # ... rest of fields
end
```

**Priority**: P1 - Correctness issue for DeepSeek models

---

### 6. Missing Synchronization Points (MEDIUM)

**Location**: Throughout `src/Model.jl`

**Issue**: Missing `oneAPI.synchronize()` calls after critical operations can cause race conditions:

```julia
# Example from GatedDeltaNet (lines 1309-1311):
copyto!(m.branch_out, reshape(Float16.(m.y_all_cpu), :, 1))
oneAPI.synchronize()  # ✓ Present here
result = mat_mul(m.ssm_out, m.branch_out)
oneAPI.synchronize()  # ✓ Present here

# But missing in other places:
# - After RMSNorm operations
# - After RoPE application
# - After KV cache updates
```

**Fix**: Add synchronization after:
1. All `copyto!` operations from CPU to GPU
2. KV cache updates before attention computation
3. Layer output before next layer input

**Priority**: P2 - Potential correctness/stability issue

---

## Medium Priority Issues

### 7. MLP GPU Implementation Uses CPU (MEDIUM)

**Location**: `src/Model.jl:507-530`

```julia
function (m::MLP)(x::oneMatrix{Float16}, cache::KVCache)
    mul!(cache.mlp_gate, m.gate_weight, x)
    mul!(cache.mlp_up, m.up_weight, x)
    
    # GPU -> CPU
    gate_cpu = Array(cache.mlp_gate)
    up_cpu = Array(cache.mlp_up)
    
    # CPU computation
    @. gate_cpu = gate_cpu * (Float32(1.0) / (Float32(1.0) + exp(-Float32(gate_cpu))))
    gate_cpu .*= up_cpu
    
    # CPU -> GPU
    copyto!(cache.mlp_gate, Float16.(gate_cpu))
    mul!(cache.branch_out, m.down_weight, cache.mlp_gate)
    
    return cache.branch_out
end
```

**Fix**: Use GPU SiLU kernel:
```julia
function (m::MLP)(x::oneMatrix{Float16}, cache::KVCache)
    mul!(cache.mlp_gate, m.gate_weight, x)
    mul!(cache.mlp_up, m.up_weight, x)
    
    # GPU SiLU activation
    silu_kernel!(cache.mlp_gate, cache.mlp_gate, length(cache.mlp_gate))
    
    # Element-wise multiply on GPU
    cache.mlp_gate .*= cache.mlp_up
    
    mul!(cache.branch_out, m.down_weight, cache.mlp_gate)
    return cache.branch_out
end
```

**Priority**: P2 - Performance issue

---

### 8. MoE GPU Implementation Uses CPU (MEDIUM)

**Location**: `src/Model.jl:558-610`

Similar to MLP, MoE routing and expert computation happens on CPU.

**Priority**: P2 - Performance issue (only affects MoE models)

---

### 9. Numerical Precision Inconsistencies (LOW-MEDIUM)

**Location**: Multiple locations

**Issue**: Mixed use of `Float32` and `Float64` for intermediate computations:

```julia
# Line 1268: Uses Float64
alpha_val = Float64(m.alpha_proj[h]) + Float64(m.ssm_dt_bias_cpu[h])
softplus_alpha = log(Float64(1.0) + exp(alpha_val))
decay = Float32(exp(softplus_alpha * Float64(m.ssm_a_cpu[h])))

# Line 1253: Uses Float32
q_norm_sq = mapreduce(v -> Float32(v)^2, +, qg)
```

**Recommendation**: Standardize on `Float32` for all intermediate computations unless `Float64` is specifically required for numerical stability.

**Priority**: P3 - Consistency issue

---

## Low Priority Issues

### 10. GC Calls Still Present in Error Paths (LOW)

**Location**: `src/Model.jl:1523-1535`, `src/Engine.jl:313-316`

While we made GC configurable in the hot path, forced GC in error handlers may still cause issues during debugging.

**Priority**: P3 - Minor issue

### 11. Preallocated Buffer Sizes Not Validated (LOW)

**Location**: `src/Model.jl:335-365` in `init_kv_cache`

Buffer sizes are fixed at initialization but not validated against actual usage.

**Priority**: P3 - Robustness issue

---

## Testing Recommendations

### Immediate Tests Required

1. **Single Layer Test**: Isolate and test each layer type (SSM, FullAttention, MLA) independently
2. **Token-by-Token Comparison**: Compare GPU vs CPU output for each token
3. **Memory Stress Test**: Load model with minimal VRAM to identify allocation issues
4. **Numerical Accuracy Test**: Compare logits between GPU and CPU backends

### Test Code Template

```julia
using Inferno, oneAPI, Test

function test_gpu_vs_cpu()
    model_path = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
    
    # Load both backends
    model_gpu, tok = load_model(model_path, backend=:gpu)
    model_cpu, _ = load_model(model_path, backend=:cpu)
    
    tokens = encode(tok, "Hello")
    
    # Compare forward pass
    caches_gpu = [Model.init_kv_cache(model_gpu.config)]
    caches_cpu = [ModelCPU.init_kv_cache_cpu(model_cpu.config, 512)]
    
    logits_gpu = Model.forward!(model_gpu, tokens, 0, caches_gpu)
    logits_cpu = ModelCPU.forward_cpu!(model_cpu, tokens, 0, caches_cpu)
    
    # Check numerical accuracy (allow 1% relative error)
    @test maximum(abs.(Array(logits_gpu) .- logits_cpu)) / maximum(abs.(logits_cpu)) < 0.01
end
```

---

## Implementation Priority

| Priority | Issue | Estimated Effort | Impact |
|----------|-------|------------------|--------|
| P0 | GPU Memory Allocation | 1 day | Critical - enables GPU |
| P1 | SSM GPU Implementation | 3 days | 10-100x speedup |
| P1 | FullAttention GPU | 2 days | Correctness + speed |
| P1 | MLA Cache Structure | 1 day | DeepSeek support |
| P2 | RoPE GPU Kernel | 1 day | 2x speedup |
| P2 | MLP GPU | 0.5 days | 2x speedup |
| P2 | Synchronization | 0.5 days | Stability |
| P3 | Numerical Consistency | 0.5 days | Code quality |

**Total Estimated Effort**: 9-10 days for complete GPU implementation

---

## Recommended Next Steps

1. **Week 1**: Fix P0 (memory allocation) and P1 (SSM GPU) issues
2. **Week 2**: Complete P1 (FullAttention, MLA) and P2 (RoPE, MLP) issues
3. **Week 3**: Testing, benchmarking, and P3 cleanup

---

## Appendix: Current Performance Characteristics

### CPU Backend (Reference)
- Token generation: ~5 tokens/second
- Memory usage: ~2GB RAM
- Numerical accuracy: Baseline

### GPU Backend (Current Broken State)
- Token generation: N/A (fails to load)
- Expected with fixes: 50-100 tokens/second
- Expected memory: ~3GB VRAM

### Target Performance
- GPU should achieve 10-20x speedup over CPU
- Memory efficiency within 20% of CPU
