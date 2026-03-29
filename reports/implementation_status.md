# Analysis of Reports - Outstanding Issues

Based on reviewing the reports in the `/reports` folder, here's a comprehensive analysis of what has been implemented and what still needs work:

## Performance Improvements Report (performance_improvements.md)

### 1. CRITICAL: Excessive CPU-GPU Data Transfers (PCIe Bottleneck)
**Status: NOT ADDRESSED**

The report identifies that:
- `GatedDeltaNet` (SSM Layer) pulls tensors to CPU for operations
- `FullAttention` softmax is computed on CPU with `collect()` calls
- Final logits are transferred to CPU for normalization

**What needs to be done:**
- Port SSM operations (1D convolution, normalization, recurrence) to GPU kernels
- Implement fused Softmax GPU kernel for attention
- Keep attention scores in VRAM during prefill

### 2. MAJOR: Naive Matrix Multiplication Kernels
**Status: PARTIALLY ADDRESSED (CPU side)**

**What was done:**
- Implemented quantized weight support for CPU inference (Q4_K, Q5_K, Q6_K, Q8_0)
- Added on-the-fly dequantization during matrix-vector multiplication

**What still needs to be done:**
- GPU kernels are still naive (no tiling, no shared memory)
- Need to integrate oneMKL for Float32 operations
- Need tiled quantized matrix multiplication kernels for GPU
- IQ2_XXS kernel needs optimization

### 3. MAJOR: Eager CPU-Side Dequantization (Memory Bloat)
**Status: PARTIALLY ADDRESSED**

**What was done:**
- Added `keep_quantized=true` option for CPU inference
- MLP weights can now stay quantized in memory
- ~19% memory savings for CPU inference

**What still needs to be done:**
- GPU path still dequantizes everything to Float32
- Need Weight-Only Quantization (WOQ) kernels for GPU
- Embeddings and attention weights still dequantized

### 4. MODERATE: CPU-Side Sampling and Logits Processing
**Status: NOT ADDRESSED**

Issues identified:
- Final logits vector (~600KB) transferred to CPU every token
- Top-P sorting is O(N log N) on CPU

**What needs to be done:**
- Move temperature scaling and argmax to GPU
- Implement GPU-based partial sort for Top-P/Top-K

### 5. MINOR: Redundant Allocations & Unoptimized Tokenizer
**Status: NOT ADDRESSED**

Issues identified:
- In-loop allocations in FullAttention prefill
- CPU broadcasts allocate new arrays
- Tokenizer uses string-heavy BPE implementation

---

## Quantization Reports

### llama_cpp_gguf_q4k_weight_storage.md
**Status: INFORMATIONAL**

This report documents how llama.cpp stores Q4_K weights. Key insights:
- Block size is 256 weights (super-block), 32 weights (sub-block)
- 144 bytes per block
- Uses integer dot product (Q4_K × Q8_K) for speed

Our implementation uses floating-point dequantization, which is simpler but potentially slower than llama.cpp's integer approach.

### qwen35_0.8b_tensor_quantization_types.md
**Status: INFORMATIONAL**

Documents which tensors get which quantization types:
- Most weights are Q4_K
- Sensitive layers (attn_v, ffn_down) get Q5_K or Q6_K
- 1D tensors (norms, biases) stay F32

**Implementation gap:**
- Our quantized weight support only covers MLP weights
- SSM tensors, attention weights, embeddings still dequantized

---

## Priority Action Items

### High Priority (Performance Critical)
1. **GPU Quantized Kernels**: Implement WOQ kernels for GPU inference
2. **Fused Softmax**: Keep attention scores on GPU
3. **SSM GPU Kernels**: Port GatedDeltaNet to GPU

### Medium Priority (Memory)
1. **Extend quantized support**: Add quantized attention weights
2. **GPU memory layout**: Store weights as raw bytes, dequantize on-the-fly

### Low Priority (Optimization)
1. **Reduce allocations**: Pre-allocate workspaces
2. **Tokenizer optimization**: Use Trie/DFA
3. **GPU sampling**: Move sampling to GPU

---

## Current Implementation Status

| Feature | CPU | GPU |
|---------|-----|-----|
| Q4_K weight storage | ✅ Implemented | ❌ Not implemented |
| Q5_K weight storage | ✅ Implemented | ❌ Not implemented |
| Q6_K weight storage | ✅ Implemented | ❌ Not implemented |
| Q8_0 weight storage | ✅ Implemented | ❌ Not implemented |
| On-the-fly dequantization | ✅ Implemented | ❌ Not implemented |
| Fused softmax | ❌ Not implemented | ❌ Not implemented |
| SSM GPU kernels | N/A | ❌ Not implemented |
| Quantized attention | ❌ Not implemented | ❌ Not implemented |
| Quantized embeddings | ❌ Not implemented | ❌ Not implemented |
