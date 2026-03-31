using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get the embedding for "The"
token = 761
x = model.embed[:, token + 1]

# Run through first 3 SSM layers
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

for i in 1:3
    global x
    layer = model.layers[i]
    x = layer(x, i-1, model.rope, caches[i])
    println("After layer $i (SSM): norm = ", round(norm(x), digits=5))
end

# Now trace through the attention layer (layer 4)
attn_layer = model.layers[4]
attn = attn_layer.op
println("\n=== Attention Layer 4 ===")
println("n_heads: ", attn.n_heads)
println("n_kv: ", attn.n_kv)
println("head_dim: ", attn.head_dim)

# Input norm
x_norm = attn_layer.in_norm(x)
println("\nAfter in_norm: ", round(norm(x_norm), digits=5))

# Attention forward
x_residual = attn_layer.op(x_norm, 3, model.rope, caches[4])
println("After attention: ", round(norm(x_residual), digits=5))
