# Systematic Weight Verification Plan

## Check each weight matrix:

1. **Embedding** (tok_embd)
   - GGUF shape: (vocab_size, hidden_size)
   - Our shape: (hidden_size, vocab_size)
   - Check: Transposition

2. **LM Head** (output)
   - GGUF shape: (vocab_size, hidden_size)
   - Our shape: (hidden_size, vocab_size)
   - Check: Transposition

3. **SSM in_proj** (ssm_in_proj)
   - GGUF shape: (conv_channels, hidden_size)
   - Our shape: Need to verify

4. **SSM gate_proj** (ssm_gate_proj)
   - GGUF shape: (d_inner, hidden_size)
   - Our shape: Need to verify

5. **SSM ssm_out** (ssm_out_proj)
   - GGUF shape: (hidden_size, d_inner)
   - Our shape: Need to verify

6. **SSM conv1d** (ssm_conv1d)
   - GGUF shape: (conv_kernel, conv_channels)
   - Our shape: Need to verify

7. **SSM alpha/beta weights**
   - GGUF shape: (num_v_heads, hidden_size)
   - Our shape: Need to verify

8. **Attention wq/wk/wv/wo**
   - Check transpositions

9. **MLP weights**
   - Check transpositions
