using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get attention layer 4
attn_layer = model.layers[4]
attn = attn_layer.op

println("=== Attention Layer 4 Dimensions ===")
println("n_heads: ", attn.n_heads)
println("n_kv: ", attn.n_kv)
println("head_dim: ", attn.head_dim)
println("wq shape: ", size(attn.wq))

# Test: What does wq * x produce?
x_test = randn(Float32, 1024)
qkv = attn.wq * x_test
println("\nqkv shape: ", size(qkv))
println("Expected: ", attn.n_heads * attn.head_dim * 2)

# Split into query and gate
q_size = attn.n_heads * attn.head_dim
query = qkv[1:q_size]
gate = qkv[q_size+1:end]

println("\nquery size: ", size(query))
println("gate size: ", size(gate))

# The reshape assumes the data is arranged as:
# [head0_all_dims, head1_all_dims, ...]
# But it might be [all_heads_dim0, all_heads_dim1, ...]

# Let's check by examining the reshape
query_reshaped = reshape(query, attn.head_dim, attn.n_heads)
println("\nquery_reshaped shape: ", size(query_reshaped))

# The question is: does reshape work correctly?
# In Julia, reshape(query, head_dim, n_heads) takes the vector and fills column by column
# So query[1:head_dim] becomes column 1, query[head_dim+1:2*head_dim] becomes column 2, etc.
# This means query_reshaped[:, h] = query[(h-1)*head_dim+1:h*head_dim]
# Which is correct if the original data is stored as [head0_all_dims, head1_all_dims, ...]
