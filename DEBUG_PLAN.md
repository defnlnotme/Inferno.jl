# Qwen3.5 CPU Inference Debug Plan

## Status: IN PROGRESS
Last updated: 2026-03-30

## Background
Model loads and runs but produces incorrect predictions:
- Expected: "\n\n" then "[Start thinking]" (matches llama.cpp)
- Actual: " th" as top prediction

## Fixes Already Applied
1. **Sigmoid vs SiLU**: Changed attention gating from SiLU to sigmoid
2. **Delta Net Operations**: Fixed sk computation, state update, and output computation
3. **Weight Transpositions**: Fixed ssm_conv1d, ssm_alpha, ssm_beta weight loading

---

## Step 1: Verify Residual Connections
Status: COMPLETED

The hidden state norm grows from ~0.57 (embedding) to 28.3 (after prompt).
This suggests residual connections may be accumulating incorrectly.

Actions:
- [x] Check if residual addition is correct in SSM layers - CORRECT
- [x] Check if residual addition is correct in Attention layers - CORRECT
- [x] Verify post_norm is applied correctly - CORRECT
- [x] Compare hidden state norm growth per layer with expected values

Findings:
- Residual connections are correct: x = x_orig + op_output
- RMS norm scales by ~32x for inputs with norm ~1 (correct behavior)
- Attention layers produce MUCH larger outputs than SSM:
  - SSM OpOut: 0.1-3.2
  - Attn OpOut: 4.5-31.7
- Final layer (24) Attention produces output with norm 31.72!
- This drives the final hidden norm to 27.89

---

## Step 2: Compare Intermediate Values with llama.cpp
Status: PENDING

Need to extract intermediate tensors from llama.cpp for comparison.

Actions:
- [ ] Modify llama.cpp to dump hidden states after each layer
- [ ] Run same prompt through both implementations
- [ ] Compare hidden state values at each layer boundary
- [ ] Identify which layer(s) diverge

---

## Step 3: Check Attention Layer Implementation
Status: COMPLETED

Attention weights have different shapes (wq: 4096x1024, wk: 512x1024).
This suggests grouped-query attention.

Actions:
- [x] Verify num_heads vs num_kv_heads handling - CORRECT (8 heads, 2 KV)
- [x] Check if Q/K/V projections are correct - CORRECT
- [x] Verify RoPE is applied correctly - CORRECT
- [x] Check softmax and attention output computation - CORRECT
- [x] Check if attention scale is correct - CORRECT (0.0625)

Findings:
- Attention outputs are larger than SSM: 4.5-31.7 vs 0.1-3.2
- This is expected behavior - attention attends to full sequence
- GQA is handled correctly (8 heads share 2 KV heads)
- Final attention output norm (8.5 for layer 4) matches manual trace

---

## Step 4: Verify MLP Layer
Status: PENDING

MLP should have gate/up/down projections with SiLU activation.

Actions:
- [ ] Check gate_proj and up_proj shapes
- [ ] Verify SiLU activation: gate * up
- [ ] Check down_proj application
- [ ] Verify residual connection after MLP

---

## Step 5: Check Final Normalization and Logits
Status: IN PROGRESS

Final norm weights are unusually large (3-4x).

Actions:
- [ ] Verify final_norm weight values match GGUF
- [ ] Check if there's a temperature/scaling applied to logits
- [ ] Compare final logits distribution with llama.cpp

---

## Step 6: Systematic Weight Verification
Status: PENDING

Verify all weight matrices are loaded with correct shapes and values.

Actions:
- [ ] Check all attention weights (wq, wk, wv, wo)
- [ ] Check all MLP weights (gate, up, down)
- [ ] Check all norm weights (input_norm, post_norm)
- [ ] Check embedding and LM head

---

## Notes

### Key Observations
- Embedding norm: 0.57
- After 3 SSM layers: 2.27
- After all 24 layers: 28.3
- Final norm output: 139.5
- Logits max: 18.5

### RMS Norm Behavior (Correct)
For input with norm N spread across D dimensions:
- rms = N / sqrt(D)
- scale = 1 / rms
- output_norm = N * scale * mean(weight) ≈ sqrt(D) * mean(weight)

For N=2.27, D=1024: scale ≈ 14.1, output ≈ 43 (matches observed)

### Next Debugging Session
Start with Step 1 - the residual connections are the most likely culprit
for the growing hidden state norm.
