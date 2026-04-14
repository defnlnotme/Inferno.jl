# Inferno.jl Development Guide

## Current Status: GGUF + Safetensors CPU Inference WORKING

Both GGUF and Safetensors inference for Qwen3.5-0.8B-VL work with coherent generation.

**Verified outputs:**
- "What is 2 + 2 ?" → "\n\n2 + 2 = 4\n\nWhat is 2 + 3 ?\n\n2 + 3 = 5..."
- Matches HuggingFace reference output exactly

---

## Roadmap

### Phase 1: Fix Safetensors CPU Inference ✓ COMPLETE

**Fixed bugs:**
1. Layer index substring matching (`"layers.1"` matched layers 10, 11, 12...)
2. Attention q_norm/k_norm missing +1 (layernorm1p convention)
3. 3D conv1d tensor handling in get_tensor()
4. Position calculation in generation loop

### Phase 2: Performance Optimizations (IN PROGRESS)

**Goal:** Speed up CPU inference 2-5x.

**Optimization targets:**
1. [ ] SIMD vectorization for matrix operations
2. [ ] LoopVectorization.jl for fused operations
3. [ ] Memory pre-allocation (reduce GC pressure)
4. [ ] BLAS optimizations (MKL vs OpenBLAS)
5. [ ] Threaded inference for attention layers
6. [ ] Cache-friendly memory layouts

**Benchmarks needed:**
- Tokens/second for various prompt lengths
- Memory usage profile
- Hot path identification (ProfileView.jl)

### Phase 3: Quantization Support

**Goal:** Support Q4, Q5, Q6, Q8 quantizations for GGUF.

**Current state:** F16 (full precision) works. Need to implement:
1. [ ] Q4_K_S / Q4_K_M dequantization
2. [ ] Q5_K_S / Q5_K_M dequantization
3. [ ] Q6_K dequantization
4. [ ] Q8_0 dequantization
5. [ ] Test each quantization level for correctness

**Reference:** llama.cpp quantization format in `ggml-common.h`

### Phase 4: Additional Model Architectures

**Goal:** Support more SSM and attention models.

**Targets:**
1. [ ] Qwen3 (non-SSM variant)
2. [ ] Mamba / Mamba-2
3. [ ] RWKV
4. [ ] Jamba (mixture of SSM and attention)

---

## Model Implementation Reference

When implementing model support, ALWAYS check the reference implementation in HuggingFace transformers first:

```
~/.local/lib/python$PYTHON_VERSION/site-packages/transformers/models/qwen3_5/
```

Key files:
- `modeling_qwen3_5.py` - Main model architecture, forward pass logic
- `configuration_qwen3_5.py` - Model configuration and hyperparameters
- `tokenization_qwen3_5.py` - Tokenizer implementation

The transformers implementation is the ground truth for:
1. Layer normalization (RMSNorm vs LayerNorm, +1 bias convention)
2. Attention patterns (RoPE, sliding window, GQA)
3. MLP structure (gate * up, then down)
4. Any architecture-specific quirks (SSM for Mamba-like models)

---

## Debugging with DaemonMode.jl

Use DaemonMode.jl to speed up debugging - avoids REPL startup overhead.

```bash
# Terminal 1: Start daemon
julia --project=. -e 'using DaemonMode; run_daemon()'

# Terminal 2: Run tests
julia --project=. -e 'using DaemonMode; run_job("tests/your_test.jl")'
```

---

## Critical: Multi-Token Verification

**NEVER claim success based on single-token tests!**

When verifying inference:
1. Generate at least 64-128 tokens
2. Check output is COHERENT TEXT (not repetition/garbage)
3. Test prompts:
   - "What is 2 + 2 ?" → Should answer "4"
   - "The capital of France" → Should say "Paris"

Single-token matching proves nothing - the model could produce garbage after first token.

---

## Known Bugs & Fixes

### Multi-token Generation Fix (Apr 2026)
**Problem:** Single-token tests passed but multi-token produced garbage.
**Fix:** KV caches must be initialized ONCE before generation loop and passed through each forward call.

WRONG:
```julia
for i in 1:64
    reset_states!(model)
    caches = [init_kv_cache(...) for _ in 1:num_layers]
    forward!(model, [token], 0, caches)  # Always pos=0!
end
```

CORRECT:
```julia
caches = [init_kv_cache(...) for _ in 1:num_layers]
logits = forward!(model, prompt_tokens, 0, caches)
pos = length(prompt_tokens)
for i in 1:64
    token = argmax(logits[:, end])
    logits = forward!(model, [token], pos, caches)  # Increment pos!
    pos += 1
end
```

### Other Fixed Bugs
1. Conv1D weight transpose: GGUF stores (C,K), kernel expects (K,C)
2. RMSNorm +1: Layer norms use layernorm1p convention (+1), SSM norms don't
3. Decay formula: `exp(ssm_a * softplus(a + dt_bias))`
4. L2 norm on Q/K: Qwen3.5 uses L2 norm, not RMSNorm, for attention
5. Safetensors BF16: dtype=3, need `UInt16 -> UInt32<<16 -> Float32`
