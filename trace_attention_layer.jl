using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Find an attention layer (1-indexed)
attention_layer_idx = 1
for (i, layer) in enumerate(model.layers)
    if !layer.is_ssm
        attention_layer_idx = i
        break
    end
end
println("First attention layer: $attention_layer_idx")

# Get the embedding for "The"
token = 761
x = model.embed[:, token + 1]

# Run through SSM layers first
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

global x
for i in 1:(attention_layer_idx-1)
    layer = model.layers[i]
    x = layer(x, i-1, model.rope, caches[i])
end

println("\nAfter SSM layers, x norm: ", round(norm(x), digits=5))

# Now trace through the attention layer
attn_layer = model.layers[attention_layer_idx]
attn = attn_layer.op
println("\n=== Attention Layer $attention_layer_idx ===")
println("n_heads: ", attn.n_heads)
println("n_kv: ", attn.n_kv)
println("head_dim: ", attn.head_dim)

# Input norm
x_norm = attn_layer.in_norm(x)
println("\nAfter in_norm: ", round(norm(x_norm), digits=5))

# Attention forward
x_residual = attn_layer.op(x_norm, attention_layer_idx-1, model.rope, caches[attention_layer_idx])
println("After attention: ", round(norm(x_residual), digits=5))
