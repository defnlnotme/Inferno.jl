# Delta Net Autoregressive Implementation Comparison

## LLAMA.CPP IMPLEMENTATION (delta-net-base.cpp)

### Input Tensors:
- q: [S_k, H_k, n_tokens, n_seqs] where n_tokens = 1
- k: [S_k, H_k, n_tokens, n_seqs]
- v: [S_v, H_v, n_tokens, n_seqs]
- g: [g_0, H_v, n_tokens, n_seqs] where g_0 = 1 or S_v
- b: [1, H_v, n_tokens, n_seqs] (beta)
- s: [S_v, S_v, H_v, n_seqs] (state)

### Step-by-step:

1. Scale Q:
   q = q * (1.0 / sqrt(S_k))

2. Permute to [S_k, n_tokens, H_k, n_seqs]:
   q = permute(q, 0, 2, 1, 3) -> [S_k, 1, H_k, n_seqs]
   k = permute(k, 0, 2, 1, 3) -> [S_k, 1, H_k, n_seqs]
   v = permute(v, 0, 2, 1, 3) -> [S_v, 1, H_v, n_seqs]

3. Reshape g and b:
   g = reshape(g, [1, g_0, H_v, n_seqs])  -> [1, 1, H_v, n_seqs] for GDA
   b = reshape(b, [1, 1, H_v, n_seqs])

4. State decay:
   g = exp(g)  -> element-wise exp
   s = s * g   -> element-wise multiplication

5. Compute sk:
   sk = s * k           -> [S_v, S_v, H_v, n_seqs] * [S_k, 1, H_k, n_seqs]
                          Note: k is broadcast to match s dimensions
                          Result: [S_v, S_v, H_v, n_seqs]
   sk = sum_rows(sk)    -> sum over first dimension
                          Result: [1, S_v, H_v, n_seqs]

6. Compute d:
   d = v - transpose(sk)  -> [S_v, 1, H_v, n_seqs] - [S_v, 1, H_v, n_seqs]
                             Result: [S_v, 1, H_v, n_seqs]
   d = d * b              -> [S_v, 1, H_v, n_seqs] * [1, 1, H_v, n_seqs]
                             Result: [S_v, 1, H_v, n_seqs]

7. Update state:
   d_t = transpose(d)     -> [1, S_v, H_v, n_seqs]
   k = repeat(k, s)       -> k broadcast to [S_v, S_v, H_v, n_seqs]
   kd = k * d_t           -> [S_v, S_v, H_v, n_seqs] * [1, S_v, H_v, n_seqs]
                             Result: [S_v, S_v, H_v, n_seqs]
   s = s + kd             -> [S_v, S_v, H_v, n_seqs]

8. Compute output:
   s_q = s * q            -> [S_v, S_v, H_v, n_seqs] * [S_k, 1, H_k, n_seqs]
                            Note: q broadcast, but H_k may differ from H_v
                            Result: [S_v, S_v, H_v, n_seqs]
   o = sum_rows(s_q)      -> [1, S_v, H_v, n_seqs]
   o = permute(o, 2, 0, 1, 3) -> [S_v, H_v, n_tokens, n_seqs]

## KEY INSIGHTS:

1. The state s is [S_v, S_v, H_v, n_seqs]
2. The operations are:
   - sk = sum_rows(s * k) where k is broadcast
   - d = (v - sk') * b
   - s = s + k * d' where k is broadcast
   - o = sum_rows(s * q) where q is broadcast

3. For GDA (Gated Delta Attention, our case):
   - H_v = H_k (same number of heads)
   - S_v = S_k (same state dimension)
   - g has shape [1, 1, H_v, n_seqs] after reshape

4. The sum_rows operation reduces over the first dimension
   - If input is [S_v, S_v, H_v, n_seqs], output is [1, S_v, H_v, n_seqs]
