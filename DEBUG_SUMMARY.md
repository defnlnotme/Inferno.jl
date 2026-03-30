# Qwen3.5 CPU Inference Debug Summary

## Fixes Applied

### 1. Sigmoid vs SiLU for Attention Gating
- Changed from SiLU (`x * sigmoid(x)`) to just `sigmoid(x)`
- File: `src/ModelCPU.jl` and `src/Model.jl`

### 2. Delta Net SSM Operations
Changed in `src/ModelCPU.jl`:
- **sk computation**: Changed from matrix-vector (`state * k`) to `k' * state`
  - This computes `sk[j] = sum_i state[i,j] * k[i]`
- **State update**: Changed from `state += d * k'` to `state += k * d'`
  - Using `BLAS.ger!(k, d, state)` instead of `BLAS.ger!(d, k, state)`
- **Output**: Changed from `state * q` to `state' * q`
  - This computes `o = sum_rows(state * q) = q' * state`

### 3. Weight Transpositions
Fixed in `src/LoaderCPU.jl`:
- `ssm_conv1d`: Julia's column-major reshape correctly interprets GGUF row-major as `(kernel, channels)`
- `ssm_alpha_weight`, `ssm_beta_weight`: Added `permutedims` to get `(num_v_heads, hidden)` shape

## Current Status
- First token prediction is correct (" " - space after "The")
- Model gets stuck generating "\n\n" tokens after the prompt
- llama.cpp correctly generates "[Start thinking]" after newlines

## Remaining Issues to Investigate

### 1. State Matrix Convention
- In llama.cpp, state has shape `[S_v, S_v, H_v, n_seqs]`
- We have `h[head_v_dim, head_k_dim, num_v_heads]` = `[128, 128, 16]`
- Need to verify `head_v_dim == head_k_dim` for all heads

### 2. Check Other Weight Transpositions
- in_proj (attn_qkv.weight)
- gate_proj (attn_gate.weight)
- ssm_out (ssm_out.weight)
- Attention weights (wq, wk, wv, wo)

### 3. Verify RoPE Implementation
- Check if multi-head RoPE is correct

### 4. Check RMSNorm Implementation
- Verify norm weights are loaded correctly

## Next Steps
1. Compare logits with llama.cpp for same prompt
2. Add debug output for intermediate tensors
3. Check each weight matrix shape systematically
