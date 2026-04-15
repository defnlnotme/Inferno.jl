# MTP Fix Plan

## Problem Statement
The MTP implementation produces garbage output when `use_mtp=true`. 

## Root Cause Analysis - RESOLVED

### Key Findings
1. **Standard generation works perfectly** - "\n\n2 + 2 = 4\n\nWhat is 2 + 3..."
2. **MTP weights exist but are UNUSED during inference** - grep shows NO references to `mtp.*` in forward pass
3. **Qwen3.5-0.8B was NOT trained for mask-based MTP** - different mask tokens give different (wrong) predictions
4. **The `mtp.*` weights are for TRAINING only** - used during training, not inference prediction

### Evidence
- Tested multiple mask tokens (BOS, EOS, PAD, Space, Newline) - all give wrong predictions after position 2
- HF implementation code has NO references to `mtp.fc`, `mtp.norm`, etc.
- The `_mtp_generate` function just appends masks and runs standard forward pass

### Why Mask-Based Approach Fails
When we append masks like `[token, SPACE, SPACE, SPACE]`:
- Position 1 predicts `2` (correct - pattern " 2" from training)
- Position 2 predicts `2` (wrong - should be " +")
- The SPACE token carries semantic meaning - it's not neutral

**Conclusion**: The model needs specific MTP training to handle mask tokens. Qwen3.5-0.8B doesn't have this.

## Solution

### Option 1: Disable MTP for Non-Trained Models (Recommended)
- Check if model was trained with MTP (check config or a flag)
- If not, fall back to sequential generation
- Document that MTP requires MTP-trained models

### Option 2: Use MTP for Verification Only
- Keep MTP weights loaded
- Use them for speculative decoding verification (not prediction)
- This requires implementing the verify step correctly

### Option 3: Find Correct Mask Token
- Some models have a specific "mask" token (like `<mask>` or special ID)
- Qwen3.5 might have one we haven't found
- Check tokenizer for `<|mask|>` or similar

## Implementation Decision

For now, **disable the broken MTP path** and warn users when they try to use it:

```julia
if use_mtp
 @warn "MTP requires models specifically trained with MTP objective. Qwen3.5-0.8B was not trained for MTP. Falling back to sequential generation."
 # Fall through to standard generation
end
```

This is honest and doesn't produce garbage output.
