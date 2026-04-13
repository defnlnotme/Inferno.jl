# Safetensors CPU Inference Status

## Status: WORKING ✓

Safetensors inference for Qwen3.5-0.8B-VL now produces correct output.

### Test Output
```
Prompt: What is 2 + 2 ?
Output:

2 + 2 = 4

What is 2 + 3 ?

2 + 3 = 5
```

This matches HuggingFace reference output exactly.

## Bugs Fixed

### 1. Layer Index Substring Matching
**Problem:** `"layers.$layer_idx"` matched unintended layers.
- `"layers.1"` matched layers 1, 10, 11, 12, ..., 19
- Layer 1+ weights were garbage (mix of multiple layers)

**Fix:** Use regex `r"layers\.$layer_idx\."` for exact matching.

### 2. Attention q_norm/k_norm Missing +1
**Problem:** Qwen uses layernorm1p convention (weight + 1).
- GGUF loader applied +1, safetensors loader didn't
- GGUF mean: 1.4268848, SF mean: 0.42688477 (diff = 1.0)

**Fix:** Add `.+ 1.0f0` when loading q_norm/k_norm weights.

### 3. 3D Conv1d Tensor Handling
**Problem:** Conv1d weights stored as [C, 1, K] in safetensors.
- Previous code didn't handle 3D tensors correctly
- Resulted in wrong weight layout

**Fix:** In `get_tensor()`, detect 3D tensors with shape[2]==1
and apply correct reshape: `reshape(data, shape[3], shape[1])'`

### 4. Position Calculation in Generation Loop
**Problem:** Wrong position passed to forward_cpu!.
- Prompt processed at positions 7-14 instead of 0-7
- Caused garbage output despite correct weights

**Fix:** Process prompt at position 0, subsequent tokens at
position `length(tokens) + i - 2`.

## Weight Verification

All weights now match between GGUF and safetensors:
- Embeddings: ✓
- Layer norms: ✓
- SSM weights (in_proj, gate_proj, out_proj, conv1d, etc.): ✓
- Attention weights (wq, wk, wv, wo, q_norm, k_norm): ✓
- MLP weights: ✓
- Final norm: ✓

## Model Details

- Architecture: Qwen3.5ForConditionalGeneration (VL model)
- Safetensors file includes visual encoder (not used for text inference)
- Weight tying: lm_head tied with embed_tokens

## Next Steps

1. Performance optimizations (Phase 2)
2. Quantization support (Phase 3)
3. Additional model architectures (Phase 4)
