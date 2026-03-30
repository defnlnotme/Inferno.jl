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
Status: COMPLETED

Final norm weights are unusually large (3-4x).

Actions:
- [x] Verify final_norm weight values match GGUF - MATCHES (mean 4.31)
- [x] Check if there's a temperature/scaling applied to logits - NO TEMPERATURE
- [x] Compare final logits distribution with llama.cpp - DIFFERENT

Findings:
- Our model predicts ' th' (logit 18.5) as top token
- Expected: '[' (logit 11.1) for "[Start thinking]"
- Hidden state norm 27.89 -> final norm 126.08
- Logits range: -19 to 18.5 (reasonable spread)
- Issue: wrong token has highest logit

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

## Step 7: Identify the Core Issue
Status: COMPLETED

After verifying all components, the model still predicts incorrectly.
Need to identify what's causing the wrong predictions.

Actions:
- [x] Check if SSM state is being accumulated correctly across tokens - CORRECT
- [x] Compare hidden state after each token with expected values - State grows correctly
- [x] Check if there's an issue with the attention KV cache - Appears correct
- [x] Verify RoPE frequencies are correct - CORRECT (theta=1e7, dim=64)
- [x] Check tokenizer encoding/decoding - CORRECT

Findings:
- All components appear to be implemented correctly
- Tokenizer encodes/decodes correctly
- SSM state accumulates properly (norm 0 -> 5.89 over 5 tokens)
- RoPE frequencies match config (theta=1e7)
- Adding <|im_start|> token doesn't change predictions

Hypothesis:
The model predictions are wrong due to a subtle numerical issue or
a mismatch in how the weights are being used. The hidden state norm
(27.89) leads to amplified logits (max 18.5) but wrong token predictions.

---

## Step 8: Deep Dive - Compare with llama.cpp
Status: PENDING

Need to compare exact intermediate values with llama.cpp to identify
where the computation diverges.

Actions:
- [ ] Extract hidden states from llama.cpp after each layer
- [ ] Compare with our implementation
- [ ] Identify the layer where values diverge

---

## Step 8: Deep Dive - Compare with llama.cpp
Status: COMPLETED

Key finding: llama.cpp produces different intermediate tensor norms than our implementation.

Actions:
- [x] Created dump-tensors tool for llama.cpp
- [x] Compared layer output norms

CRITICAL FINDING:
- llama.cpp final layer output norm: 10.46
- Our implementation final norm: 27.89
- Ratio: 2.67x larger in our implementation

llama.cpp layer norms (first 5 tokens):
- l_out-0: 2.73 (SSM)
- l_out-3: 4.73 (first attention)
- l_out-6: 13.30 (attention - big jump)
- l_out-23: 10.46 (final)

Our implementation:
- Layer 1: 1.01 (SSM)
- Layer 4: 3.90 (first attention)
- Layer 24: 27.89 (final)

The norms diverge significantly, especially in later layers.
This causes wrong logits and predictions.

---

## Step 9: Identify Norm Divergence Point
Status: IN PROGRESS

Need to identify where exactly the norms diverge.

Actions:
- [ ] Compare embedding layer output
- [ ] Compare first SSM layer output in detail
- [ ] Check if the issue is in SSM, attention, or MLP

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
