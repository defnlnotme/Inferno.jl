# Delta Net Implementation Comparison

## Key Differences Found

### 1. State Storage Layout

**llama.cpp:**
```cpp
// state is stored transposed: s_out[j*S_v + i] = S[i][j]
// so row j of s_out = column j of S (contiguous access)
float * s_out = state_out_base + (iv3 * H + iv1) * S_v * S_v;
```

**Our implementation:**
```julia
# State is stored as (head_v_dim, head_v_dim, num_v_heads) = (128, 128, 16)
state = view(ssm.h, :, :, h)  # (128, 128)
```

The state layout might be transposed! Let me verify.

### 2. Decay Application

**llama.cpp (KDA mode - Key-Value Decay Attention):**
```cpp
if (kda) {
    // precompute exp(g) into delta scratch
    for (int64_t i = 0; i < S_v; ++i) {
        delta[i] = expf(g_d[i]);
    }
    // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
    for (int64_t j = 0; j < S_v; ++j) {
        ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
    }
}
```

**Our implementation:**
```julia
state .*= decay  # Single scalar decay per head
```

CRITICAL: llama.cpp uses PER-ROW decay (exp(g) where g is a vector),
while we use a single scalar decay per head!

### 3. State Update

**llama.cpp:**
```cpp
// delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
for (int64_t j = 0; j < S_v; ++j) {
    float sum = 0.0f;
    ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
    delta[j] = (v_d[j] - sum) * beta_val;
}

// outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
for (int64_t j = 0; j < S_v; ++j) {
    ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
}
```

**Our implementation:**
```julia
# sk = k' * state
sk = k_normalized' * state

# d = beta * (v - sk)
d = beta .* (vg .- vec(sk))

# state += k * d'
BLAS.ger!(1.0f0, k_normalized, d, state)
```

This looks equivalent but the STATE LAYOUT may be transposed!

### 4. Output Computation

**llama.cpp:**
```cpp
// attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
for (int64_t j = 0; j < S_v; ++j) {
    float sum = 0.0f;
    ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
    attn_data[j] = sum * scale;
}
```

**Our implementation:**
```julia
yg = state' * q_normalized
```

Again, the state layout matters here!

## Next Steps

1. Check if our state is transposed compared to llama.cpp
2. Check if we're using per-row decay vs scalar decay
3. Verify the RMS norm implementation
