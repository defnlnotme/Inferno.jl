# Qwen3.5-0.8B SSM/Attention CPU Inference Debug Summary

## Status: MULTI-TOKEN GENERATION NOW WORKING (Apr 2026)

**BREAKTHROUGH**: The SSM state accumulation bug has been fixed!

### The Fix
The issue was improper KV cache management during multi-token generation.

**WRONG** (caused garbage output):
```julia
# Reset and reinitialize inside the loop - WRONG!
for i in 1:64
    Inferno.ModelCPU.reset_states_cpu!(model)
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    logits = Inferno.ModelCPU.forward_cpu!(model, [next_token], 0, caches)
end
```

**CORRECT** (produces coherent text):
```julia
# Initialize caches ONCE outside the loop, accumulate state
Inferno.ModelCPU.reset_states_cpu!(model)
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

# Process prompt first
logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches)

for i in 1:64
    next_token = argmax(logits[:, end])
    # Forward with accumulated state - pass current_pos and existing caches
    logits = Inferno.ModelCPU.forward_cpu!(model, [next_token], current_pos, caches)
    current_pos += 1
end
```

### Verified Output

**Test 1**: "What is 2 + 2 ?"
```
2 + 2 = 4
What is 2 + 3 ?
2 + 3 = 5
What is 2 + 4 ?
2 + 4 = 6
...
```

**Test 2**: "The capital of France is"
```
Paris.
The capital of France is Paris.
...
```

Both tests show coherent, contextually appropriate completions!

---

## Previously Fixed Bugs

### 1. SSM State Storage (V, K) vs (K, V)
- SSM state is stored as (V, K) in Julia, transposed from HF's (K, V)
- Correct ops: `sk = state * k`, `state .+= d .* k'`, `y = state * q`
- NOT: `sk = state' * k`, `state .+= k .* d'`, `y = state' * q`

### 2. RMSNorm +1 Convention
- RMSNorm weights need +1 (layernorm1p convention) for layer norms
- SSM norms do NOT use +1

### 3. Decay Formula
- Was: `exp(softplus*ssm_a)` 
- Should be: `exp(ssm_a * softplus(a+dt_bias))`
- Decay: `g = ssm_a * softplus(a + dt_bias)`, then `exp(g)`

### 4. Conv1d Weight Transpose
- GGUF stores (C, K) but kernel expects (K, C) - transpose needed

### 5. Float Literal syntax
- `1e-6f0` should be `1.0f-6` in Julia

### 6. L2 Norm on Q/K
- Qwen3.5 uses L2 norm on Q/K (not RMSNorm!) - critical for attention

### 7. Gated Norm
- Output: `norm(output) * silu(z)` at end of SSM block

### 8. Matrix Convention
- GGUF stores (out, in), Julia needs (in, out) after transform

### 9. Safetensors BF16
- BF16 dtype=3, need `UInt16 -> UInt32<<16 -> Float32`

---

## Key Lessons

1. **Single-token tests are insufficient** - They can pass while multi-token fails
2. **State accumulation is critical** - Must preserve KV cache across tokens
3. **Position matters** - Each forward call needs correct position offset
4. **Compare against HF reference** - Transformers implementation is ground truth

---

## Test Files

- `tests/reference/julia/test_multitoken_v3.jl` - Working multi-token test
- `tests/reference/julia/test_france.jl` - Capital of France test
- `tests/reference/test_ssm_against_hf.py` - SSM ops comparison
- `tests/reference/test_attention_against_hf.py` - Attention ops comparison
