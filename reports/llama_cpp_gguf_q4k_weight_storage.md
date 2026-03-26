# How llama.cpp Stores and Uses Q4_K GGUF Weights

**Date**: 2026-03-26  
**Source**: llama.cpp repository at `~/dev/models/llama.cpp-setup/llama.cpp`  
**Relevant files**: `ggml/include/gguf.h`, `ggml/src/gguf.cpp`, `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/quants.c`, `src/llama-model-loader.cpp`, `src/llama-quant.cpp`

---

## 1. Clarification: Q4_K_XL Does Not Exist

There is no `Q4_K_XL` quantization type in llama.cpp. The available Q4_K variants are:

- **Q4_K_S** (Small, `LLAMA_FTYPE_MOSTLY_Q4_K_S = 14`): Mostly Q4_K, with some critical attention/FFN layers upgraded to Q5_K.
- **Q4_K_M** (Medium, `LLAMA_FTYPE_MOSTLY_Q4_K_M = 15`): Mostly Q4_K, with more aggressive upgrades to Q5_K/Q6_K for sensitive layers.

Both variants use the identical underlying `block_q4_K` structure. The S/M distinction is a *file-level quantization strategy* that decides which tensors get higher precision, not a different block format.

---

## 2. GGUF File Format

### 2.1 File Layout

```
Offset  Size          Description
------  ----          -----------
0       4 bytes       Magic: "GGUF" (0x47 0x47 0x55 0x46)
4       4 bytes       Version (uint32_t, currently 3)
8       8 bytes       Number of tensors (int64_t)
16      8 bytes       Number of KV metadata pairs (int64_t)
24      variable      Key-value metadata pairs
  +variable           Tensor info entries (name, shape, type, offset)
  +variable           Padding to alignment boundary
  +variable           Tensor data binary blob
```

### 2.2 Metadata Key-Value Pairs

Each KV pair consists of:
1. **Key**: length-prefixed UTF-8 string (no null terminator)
2. **Value type**: int32 enum
3. **Value data**: binary representation (arrays include element type + count)

Important metadata keys:
- `general.alignment` (uint32): alignment for tensor data (default 32, must be power of 2)

### 2.3 Tensor Info (per tensor)

1. **Name**: string
2. **n_dims**: uint32 (up to `GGML_MAX_DIMS=4`)
3. **Shape**: int64 for each dimension
4. **Type**: int32 enum (`GGML_TYPE_Q4_K = 12`)
5. **Offset**: uint64 offset from start of tensor data blob

### 2.4 Tensor Data

A single contiguous binary blob, aligned to `general.alignment`. Each tensor's data is padded to the alignment boundary. Tensor offsets are relative to the start of this blob.

---

## 3. Q4_K Block Structure

### 3.1 block_q4_K Definition

From `ggml/src/ggml-common.h:302-317`:

```c
#define QK_K 256          // super-block size
#define K_SCALE_SIZE 12   // bytes for packed scales

typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales (FP16)
            ggml_half dmin; // super-block scale for quantized mins (FP16)
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // 8 scales + 8 mins, quantized to 6 bits, packed
    uint8_t qs[QK_K/2];           // 256 x 4-bit quantized weights (128 bytes)
} block_q4_K;
```

**Size**: `2 × sizeof(ggml_half) + 12 + 128 = 4 + 12 + 128 = 144 bytes`

### 3.2 Block Anatomy

| Field | Size | Purpose |
|-------|------|---------|
| `d` | 2 bytes (FP16) | Super-block scale for the 8 sub-block scales |
| `dmin` | 2 bytes (FP16) | Super-block scale for the 8 sub-block mins |
| `scales[12]` | 12 bytes | 8 scales + 8 mins packed as 6-bit values |
| `qs[128]` | 128 bytes | 256 quantized weights at 4 bits each (nibble-packed) |

- **Super-block**: 256 weights
- **Sub-blocks**: 8 sub-blocks of 32 weights each
- **Effective bits per weight**: 144 × 8 / 256 = **4.5 bpw**
- **Quantization formula**: `x = (d × sub_scale) × q − (dmin × sub_min)`

### 3.3 Packed Scales Layout

The 12-byte `scales` array encodes 8 scale values and 8 min values, each quantized to 6 bits (64 levels). Packing is done via `get_scale_min_k4()` in `ggml-quants.c`:

```
scales[0..3]  : sc[0..3] lower 6 bits
scales[4..7]  : m[0..3]  lower 6 bits
scales[8..11] : interleaved upper 2 bits of sc[4..7] and m[4..7]
```

Unpacking logic:
```c
if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
} else {
    *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
    *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
}
```

### 3.4 Quantized Weight Packing

The 128-byte `qs` array stores 256 4-bit values (0–15), packed two per byte in nibbles. The layout processes 64 weights at a time (4 groups total):
- First 32 weights: low nibbles of 32 bytes
- Next 32 weights: high nibbles of the same 32 bytes

---

## 4. Memory Layout During Inference

### 4.1 Loading

Two primary loading modes:

1. **mmap (default)**: The GGUF file is memory-mapped directly. Tensor data pointers point into the mmap region. No explicit copy occurs until pages are accessed:
   ```c
   uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;
   ggml_backend_tensor_alloc(buf_mmap, cur, data);
   ```

2. **Direct I/O / GPU upload**: Data is read from file into pinned host memory, then uploaded to GPU memory via async transfers.

### 4.2 Tensor Dimensions (Llama/Qwen-style architecture)

```
tok_embd:     [n_embd, n_vocab]            -- typically F16, not quantized
attn_norm:    [n_embd]                     -- F32, never quantized (1D)
wq (query):   [n_embd, n_embd_head * n_head]
wk (key):     [n_embd, n_embd_head * n_head_kv]
wv (value):   [n_embd, n_embd_head * n_head_kv]
wo (output):  [n_embd_head * n_head, n_embd]
ffn_norm:     [n_embd]                     -- F32, never quantized (1D)
ffn_gate:     [n_embd, n_ff]
ffn_down:     [n_ff, n_embd]
ffn_up:       [n_embd, n_ff]
output_norm:  [n_embd]                     -- F32, never quantized (1D)
output:       [n_embd, n_vocab]            -- may be quantized or F16
```

All 2D weight tensors are quantized (column dimension must be divisible by `QK_K=256`). 1D tensors (norms, biases, RoPE factors) are kept in F32 or F16.

### 4.3 File-level Memory Layout

For a Q4_K tensor with shape `[n_rows, n_cols]` where `n_cols % 256 == 0`:

```
block_q4_K[0]           -- row 0, columns 0..255      (144 bytes)
block_q4_K[1]           -- row 0, columns 256..511    (144 bytes)
...
[padding to alignment]
block_q4_K[n_cols/256]  -- row 1, columns 0..255      (144 bytes)
...
```

- `nb[0] = 144` (type_size for Q4_K)
- `nb[1] = 144 × (n_cols / 256)` bytes per row

---

## 5. Inference: Dot Product Without Full Dequantization

### 5.1 The Key Optimization

During matrix multiplication, weights (Q4_K) are NOT fully dequantized to FP32. Instead, activations are quantized to Q8_K, and the dot product is computed primarily in **integer arithmetic**:

```
dot = (d_q4 × d_q8) × Σ(scale_i × q4_i × q8_i) − (dmin_q4 × d_q8) × Σ(min_i × q8_sums_i)
```

This is implemented in `ggml_vec_dot_q4_K_q8_K()` (in `ggml/src/ggml-cpu/quants.c:590-663`).

### 5.2 Step-by-step Process

For each super-block of 256 weights:

1. **Unpack 4-bit quants** to int8 from nibble-packed `qs[]`
2. **Unpack scales** from the 6-bit packed `scales[]` array using bit manipulation
3. **Integer accumulation**: For each sub-block of 32 weights:
   ```
   accum += scale × q4_weight × q8_activation
   min_accum += min × q8_block_sums
   ```
4. **Apply floating-point scales** at the super-block level:
   ```
   result += (d_q4 × d_q8) × accum − (dmin_q4 × d_q8) × min_accum
   ```

### 5.3 Why This Is Fast

- Integer multiply-accumulate is much faster than FP32 on most hardware
- The activation quantization (to Q8_K) happens once per row, amortized across all dot products
- The 6-bit sub-block scales provide enough dynamic range to maintain accuracy
- SIMD instructions (AVX2, AVX-512, ARM NEON) accelerate the inner loops

---

## 6. Layer-Specific Quantization Strategies

### 6.1 Q4_K_S (Small)

| Tensor Category | Quantization | Notes |
|-----------------|--------------|-------|
| `attn_v` | Q5_K | First 4 layers upgraded |
| `attn_q`, `attn_k` | Q4_K | Standard |
| `attn_output` | Q4_K | Standard |
| `ffn_down` | Q5_K | First 1/8 of layers upgraded |
| `ffn_gate`, `ffn_up` | Q4_K | Standard |
| 1D tensors | F32/F16 | Never quantized |

### 6.2 Q4_K_M (Medium)

| Tensor Category | Quantization | Notes |
|-----------------|--------------|-------|
| `attn_v` | Q6_K | Higher precision for value weights |
| `attn_qkv` | Q5_K | If present in architecture |
| `attn_q`, `attn_k` | Q4_K | Standard |
| `attn_output` | Q5_K | Upgraded from Q4_K |
| `ffn_down` | Q6_K | First and last 1/16 of layers use Q6_K; middle layers may use Q4_K |
| `ffn_gate`, `ffn_up` | Q4_K | Standard |
| 1D tensors | F32/F16 | Never quantized |

### 6.3 Intuition

Attention value projection (`wv`) and FFN down projection (`ffn_down`) are the most sensitive to quantization errors. Errors in these weights compound through the residual stream. The S/M variants allocate extra bits to these layers while keeping less sensitive layers at Q4_K to maintain an average of ~4.5 bpw.

---

## 7. Summary

| Property | Value |
|----------|-------|
| Block size | 256 weights (super-block), 32 weights (sub-block) |
| Bits per weight | ~4.5 |
| Quantized values | 4-bit (16 levels) |
| Sub-block metadata | 6-bit scales and mins (64 levels each) |
| Super-block metadata | 2 × FP16 (global scale and min) |
| Block struct size | 144 bytes |
| Inference strategy | Integer dot product (Q4_K × Q8_K) |
| Quantization formula | `x = (d × scale) × q − (dmin × min)` |
| Sensitive layers | `attn_v`, `ffn_down` get higher precision in S/M variants |
| 1D tensors | Always F32 or F16, never quantized |
