# Qwen3.5 0.8B — Complete Tensor List with Quantization Types

**Date**: 2026-03-26  
**Source**: llama.cpp at `~/dev/models/llama.cpp-setup/llama.cpp`  
**Architecture**: `LLM_ARCH_QWEN35` (hybrid full-attention + linear-attention)  
**Quantization strategies**: Q4_K_S, Q4_K_M

---

## 1. Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_embd` | 1024 |
| `n_layer` | 24 |
| `n_vocab` | 151936 |
| `n_head` | 8 |
| `n_head_kv` | 2 |
| `head_dim` | 256 |
| `n_rot` | 64 (partial rotary, 25%) |
| `n_ff` | 3584 |
| `full_attention_interval` | 4 |
| `ssm_d_inner` | 2048 |
| `ssm_d_state` | 128 |
| `ssm_d_conv` | 4 |
| `ssm_dt_rank` | 16 |

**Layer distribution**: Every 4th layer (3, 7, 11, 15, 19, 23) is full self-attention; the remaining 18 layers are linear attention (Gated DeltaNet).

---

## 2. Quantization Type Legend

| Type | Bits/weight | Description |
|------|-------------|-------------|
| **Q4_K** | ~4.5 | 4-bit quantized weights, 6-bit sub-block scales, FP16 super-block scales |
| **Q5_K** | ~5.5 | 5-bit quantized weights, 6-bit sub-block scales |
| **Q6_K** | ~6.6 | 6-bit quantized weights, 8-bit scales |
| **F32** | 32 | Full precision float (1D tensors, never quantized) |
| **F16** | 16 | Half precision float |

**Upgrade logic**:
- `use_more_bits(i, n)` returns `true` when layer `i` is in the first 1/8, last 1/8, or every 3rd layer in between. For `n_layer=24`: layers 0, 1, 2, 5, 8, 11, 14, 17, 20, 21, 22, 23.

---

## 3. Global Tensors

| # | Tensor Name | Shape | Q4_K_S | Q4_K_M |
|---|-------------|-------|--------|--------|
| 1 | `token_embd.weight` | `[1024, 151936]` | Q4_K | Q4_K |
| 2 | `output_norm.weight` | `[1024]` | F32 | F32 |
| 3 | `output.weight` | `[1024, 151936]` | Q6_K | Q6_K |

**Note**: If embeddings are tied (no separate `output.weight`), `token_embd.weight` gets Q6_K instead.

---

## 4. Full Attention Layers (indices 3, 7, 11, 15, 19, 23)

Each full-attention layer has these tensors:

| # | Tensor Name | Shape | Q4_K_S | Q4_K_M |
|---|-------------|-------|--------|--------|
| 1 | `blk.{i}.attn_norm.weight` | `[1024]` | F32 | F32 |
| 2 | `blk.{i}.attn_q.weight` | `[1024, 4096]` | Q4_K | Q4_K |
| 3 | `blk.{i}.attn_q_norm.weight` | `[256]` | F32 | F32 |
| 4 | `blk.{i}.attn_k.weight` | `[1024, 512]` | Q4_K | Q4_K |
| 5 | `blk.{i}.attn_k_norm.weight` | `[256]` | F32 | F32 |
| 6 | `blk.{i}.attn_v.weight` | `[1024, 512]` | see below | see below |
| 7 | `blk.{i}.attn_output.weight` | `[2048, 1024]` | Q4_K | Q4_K |
| 8 | `blk.{i}.post_attention_norm.weight` | `[1024]` | F32 | F32 |
| 9 | `blk.{i}.ffn_gate.weight` | `[1024, 3584]` | Q4_K | Q4_K |
| 10 | `blk.{i}.ffn_down.weight` | `[3584, 1024]` | see below | see below |
| 11 | `blk.{i}.ffn_up.weight` | `[1024, 3584]` | Q4_K | Q4_K |

### Full attention `attn_v` breakdown (6 layers total):

| Layer | Q4_K_S | Q4_K_M |
|-------|--------|--------|
| blk.3 | Q5_K (i<4) | Q4_K (3 not in use_more_bits set for 6 attn_v) |
| blk.7 | Q4_K | Q4_K |
| blk.11 | Q4_K | Q4_K |
| blk.15 | Q4_K | Q4_K |
| blk.19 | Q4_K | Q4_K |
| blk.23 | Q4_K | Q4_K |

### Full attention `ffn_down` breakdown:

| Layer | Q4_K_S | Q4_K_M |
|-------|--------|--------|
| blk.3 | Q5_K (3<24/8=3, false → but `i_layer < n_layer/8` truncates to 3, so 3≥3 → Q4_K) | Q4_K (3 not in use_more_bits for 24 layers) |
| blk.7 | Q4_K | Q4_K |
| blk.11 | Q4_K | Q4_K |
| blk.15 | Q4_K | Q4_K |
| blk.19 | Q4_K | Q4_K |
| blk.23 | Q4_K | Q4_K |

---

## 5. Linear Attention Layers (indices 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22)

Each linear-attention layer has these tensors:

| # | Tensor Name | Shape | Q4_K_S | Q4_K_M |
|---|-------------|-------|--------|--------|
| 1 | `blk.{i}.attn_norm.weight` | `[1024]` | F32 | F32 |
| 2 | `blk.{i}.attn_qkv.weight` | `[1024, 6144]` | see below | see below |
| 3 | `blk.{i}.attn_gate.weight` | `[1024, 2048]` | Q4_K | Q4_K |
| 4 | `blk.{i}.ssm_conv1d.weight` | `[4, 6144]` | F32 (1D, excluded) | F32 (1D, excluded) |
| 5 | `blk.{i}.ssm_dt.bias` | `[16]` | F32 (1D/bias, excluded) | F32 (1D/bias, excluded) |
| 6 | `blk.{i}.ssm_a` | `[16]` | F32 (1D, excluded) | F32 (1D, excluded) |
| 7 | `blk.{i}.ssm_beta.weight` | `[1024, 16]` | F32 (column dim 16 < 256, cannot quantize) | F32 (column dim 16 < 256, cannot quantize) |
| 8 | `blk.{i}.ssm_alpha.weight` | `[1024, 16]` | F32 (column dim 16 < 256, cannot quantize) | F32 (column dim 16 < 256, cannot quantize) |
| 9 | `blk.{i}.ssm_norm.weight` | `[128]` | F32 (1D, excluded) | F32 (1D, excluded) |
| 10 | `blk.{i}.ssm_out.weight` | `[2048, 1024]` | Q4_K | Q4_K |
| 11 | `blk.{i}.post_attention_norm.weight` | `[1024]` | F32 | F32 |
| 12 | `blk.{i}.ffn_gate.weight` | `[1024, 3584]` | Q4_K | Q4_K |
| 13 | `blk.{i}.ffn_down.weight` | `[3584, 1024]` | see below | see below |
| 14 | `blk.{i}.ffn_up.weight` | `[1024, 3584]` | Q4_K | Q4_K |

### Linear attention `attn_qkv` breakdown (18 layers, sequential attn_v index 0–17):

The `attn_qkv` tensor matches the `ATTENTION_QKV` category. In Q4_K_M it also gets the `category_is_attn_v` upgrade path. The attn_v sequential index runs 0–17 across these 18 layers.

| Layer | Attn_v idx | Q4_K_S | Q4_K_M |
|-------|------------|--------|--------|
| blk.0 | 0 | Q5_K (0<4) | Q4_K |
| blk.1 | 1 | Q5_K (1<4) | Q4_K |
| blk.2 | 2 | Q5_K (2<4) | Q4_K |
| blk.4 | 3 | Q5_K (3<4) | Q4_K |
| blk.5 | 4 | Q4_K | Q4_K |
| blk.6 | 5 | Q4_K | Q4_K |
| blk.8 | 6 | Q4_K | Q4_K |
| blk.9 | 7 | Q4_K | Q4_K |
| blk.10 | 8 | Q4_K | Q4_K |
| blk.12 | 9 | Q4_K | Q4_K |
| blk.13 | 10 | Q4_K | Q4_K |
| blk.14 | 11 | Q4_K | Q4_K |
| blk.16 | 12 | Q4_K | Q4_K |
| blk.17 | 13 | Q4_K | Q4_K |
| blk.18 | 14 | Q4_K | Q4_K |
| blk.20 | 15 | Q4_K | Q4_K |
| blk.21 | 16 | Q4_K | Q4_K |
| blk.22 | 17 | Q4_K | Q4_K |

### Linear attention `ffn_down` breakdown (18 layers, layer index 0–22):

| Layer | Q4_K_S | Q4_K_M |
|-------|--------|--------|
| blk.0 | Q5_K (0<3) | Q6_K (0 in first 1/8) |
| blk.1 | Q5_K (1<3) | Q6_K (1 in first 1/8) |
| blk.2 | Q5_K (2<3) | Q6_K (2 in first 1/8) |
| blk.4 | Q4_K | Q4_K |
| blk.5 | Q4_K | Q6_K (5 in use_more_bits) |
| blk.6 | Q4_K | Q4_K |
| blk.8 | Q4_K | Q6_K (8 in use_more_bits) |
| blk.9 | Q4_K | Q4_K |
| blk.10 | Q4_K | Q4_K |
| blk.12 | Q4_K | Q6_K (12 in use_more_bits) |
| blk.13 | Q4_K | Q4_K |
| blk.14 | Q4_K | Q4_K |
| blk.16 | Q4_K | Q6_K (16 in use_more_bits) |
| blk.17 | Q4_K | Q4_K |
| blk.18 | Q4_K | Q4_K |
| blk.20 | Q4_K | Q6_K (20 in use_more_bits) |
| blk.21 | Q4_K | Q6_K (21 in last 1/8) |
| blk.22 | Q4_K | Q6_K (22 in last 1/8) |

---

## 6. Complete Per-Layer Summary Table

### Full Attention Layers

| Layer | attn_norm | attn_q | attn_q_norm | attn_k | attn_k_norm | attn_v | attn_output | post_attn_norm | ffn_gate | ffn_down | ffn_up |
|-------|-----------|--------|-------------|--------|-------------|--------|-------------|----------------|----------|----------|--------|
| blk.3 | F32 | Q4_K | F32 | Q4_K | F32 | **Q5_K** / Q4_K | Q4_K | F32 | Q4_K | Q4_K / Q4_K | Q4_K |
| blk.7 | F32 | Q4_K | F32 | Q4_K | F32 | Q4_K | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.11 | F32 | Q4_K | F32 | Q4_K | F32 | Q4_K | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.15 | F32 | Q4_K | F32 | Q4_K | F32 | Q4_K | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.19 | F32 | Q4_K | F32 | Q4_K | F32 | Q4_K | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.23 | F32 | Q4_K | F32 | Q4_K | F32 | Q4_K | Q4_K | F32 | Q4_K | Q4_K | Q4_K |

Format: Q4_K_S / Q4_K_M (where they differ)

### Linear Attention Layers

| Layer | attn_norm | attn_qkv | attn_gate | ssm_conv1d | ssm_dt.bias | ssm_a | ssm_beta | ssm_alpha | ssm_norm | ssm_out | post_attn_norm | ffn_gate | ffn_down | ffn_up |
|-------|-----------|----------|-----------|------------|-------------|-------|----------|-----------|----------|---------|----------------|----------|----------|--------|
| blk.0 | F32 | **Q5_K** / Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | **Q5_K** / **Q6_K** | Q4_K |
| blk.1 | F32 | **Q5_K** / Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | **Q5_K** / **Q6_K** | Q4_K |
| blk.2 | F32 | **Q5_K** / Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | **Q5_K** / **Q6_K** | Q4_K |
| blk.4 | F32 | **Q5_K** / Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.5 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K / **Q6_K** | Q4_K |
| blk.6 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.8 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K / **Q6_K** | Q4_K |
| blk.9 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.10 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.12 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K / **Q6_K** | Q4_K |
| blk.13 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.14 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.16 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K / **Q6_K** | Q4_K |
| blk.17 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.18 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K | Q4_K |
| blk.20 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K / **Q6_K** | Q4_K |
| blk.21 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K / **Q6_K** | Q4_K |
| blk.22 | F32 | Q4_K | Q4_K | F32 | F32 | F32 | F32 | F32 | F32 | Q4_K | F32 | Q4_K | Q4_K / **Q6_K** | Q4_K |

Format: Q4_K_S / Q4_K_M (where they differ). All SSM tensors are F32 because their column dimensions are too small for block quantization (QK_K=256 required).

---

## 7. Tensor Count Summary

| Category | Quantized (Q4_K) | Upgraded (Q5_K/Q6_K) | F32 (excluded) | Total |
|----------|------------------|----------------------|----------------|-------|
| Global | 1 | 1 (output) | 1 (output_norm) | 3 |
| Per-layer norms | 0 | 0 | 48 (2 per layer × 24) | 48 |
| Full attn (6 layers) | 48 | 1 (attn_v blk.3 in Q4_K_S) | 12 (q_norm + k_norm) | 61 |
| Linear attn (18 layers) | 108 | varies | 90 (SSM + norms) | 198+ |
| FFN (24 layers) | 72 | varies (ffn_down) | 0 | 72 |
| **Total** | **~229** | **~3–12** | **~151** | **~382** |

---

## 8. Source References

| File | Lines | Content |
|------|-------|---------|
| `src/llama-arch.cpp` | 1029–1054 | QWEN35 tensor name list |
| `src/llama-arch.cpp` | 337–409 | Tensor name-to-string mapping |
| `src/llama-model.cpp` | 7258–7313 | QWEN35 tensor creation with shapes |
| `src/llama-model.cpp` | 2399–2426 | QWEN35 hyperparameter loading |
| `src/llama-quant.cpp` | 116–151 | Tensor category classification |
| `src/llama-quant.cpp` | 412–414 | `use_more_bits` helper |
| `src/llama-quant.cpp` | 529–542 | attn_v quantization logic |
| `src/llama-quant.cpp` | 589–598 | ffn_down quantization logic |
| `ggml/src/ggml-common.h` | 302–317 | `block_q4_K` struct definition |
| `ggml/src/ggml-quants.c` | 1412–1434 | Q4_K dequantization |
| `ggml/src/ggml-cpu/quants.c` | 590–663 | Q4_K × Q8_K dot product |
