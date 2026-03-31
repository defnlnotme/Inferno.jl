using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get "The" embedding
token = 761
x = model.embed[:, token + 1]

# Run through first 3 SSM layers
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

for i in 1:3
    global x
    x = model.layers[i](x, i-1, model.rope, caches[i])
end

println("After 3 SSM layers, x norm: ", round(norm(x), digits=5))

# Now trace attention layer 4
attn_layer = model.layers[4]
attn = attn_layer.op

# Input norm
x_norm = attn_layer.in_norm(x)
println("\nAfter in_norm: ", round(norm(x_norm), digits=5))

# Q, K, V projections
qkv = attn.wq * x_norm
k = attn.wk * x_norm
v = attn.wv * x_norm

println("\n=== After projections ===")
println("qkv norm: ", round(norm(qkv), digits=5))
println("k norm: ", round(norm(k), digits=5))
println("v norm: ", round(norm(v), digits=5))

# Split qkv
q_size = attn.n_heads * attn.head_dim
query = qkv[1:q_size]
gate = qkv[q_size+1:end]

println("\nquery norm: ", round(norm(query), digits=5))
println("gate norm: ", round(norm(gate), digits=5))

# Reshape
query_states = reshape(query, attn.head_dim, attn.n_heads)
k_reshaped = reshape(k, attn.head_dim, attn.n_kv)
v_reshaped = reshape(v, attn.head_dim, attn.n_kv)

println("\n=== After reshape ===")
println("query_states shape: ", size(query_states))
println("k_reshaped shape: ", size(k_reshaped))
println("v_reshaped shape: ", size(v_reshaped))

# Q/K normalization
for h in 1:attn.n_heads
    q_h = view(query_states, :, h)
    Inferno.ModelCPU.rmsnorm_cpu!(q_h, q_h, attn.q_norm)
end
for h in 1:attn.n_kv
    k_h = view(k_reshaped, :, h)
    Inferno.ModelCPU.rmsnorm_cpu!(k_h, k_h, attn.k_norm)
end

println("\nAfter Q/K norm:")
println("query_states norm: ", round(norm(query_states), digits=5))
println("k_reshaped norm: ", round(norm(k_reshaped), digits=5))

# Apply RoPE (position 3)
Inferno.ModelCPU.apply_rotary_emb!(query_states, 3, model.rope)
Inferno.ModelCPU.apply_rotary_emb!(k_reshaped, 3, model.rope)

println("\nAfter RoPE:")
println("query_states norm: ", round(norm(query_states), digits=5))
println("k_reshaped norm: ", round(norm(k_reshaped), digits=5))

# Update KV cache
Inferno.ModelCPU.update_kv_cache!(caches[4], k_reshaped, v_reshaped, 3)

println("\nKV cache updated at position 3")

# Compute attention scores for head 1
println("\n=== Attention Scores Head 1 ===")
q_h = query_states[:, 1]
K_h = view(caches[4].k, :, 1, 1:4)  # positions 0-3
V_h = view(caches[4].v, :, 1, 1:4)

scores = K_h' * q_h
scores .*= attn.scale

println("scores: ", round.(scores, digits=3))

# Softmax
max_score = maximum(scores)
scores_softmax = exp.(scores .- max_score)
scores_softmax ./= sum(scores_softmax)

println("softmax scores: ", round.(scores_softmax, digits=3))
