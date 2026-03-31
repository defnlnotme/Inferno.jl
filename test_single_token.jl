using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get "The" embedding
token = 761
x = model.embed[:, token + 1]

# Initialize caches
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

# Run through all layers
for (i, layer) in enumerate(model.layers)
    global x
    x = layer(x, i-1, model.rope, caches[i])
end

# Final norm
x_final = model.final_norm(x)

# LM head
logits = model.lm_head * x_final

println("=== Single Token 'The' Forward Pass ===")
println("Final hidden norm: ", round(norm(x_final), digits=5))
println("Logits norm: ", round(norm(logits), digits=5))

# Check top tokens
top_k = 10
top_indices = sortperm(vec(logits), rev=true)[1:top_k]
println("\nTop $top_k tokens:")
for idx in top_indices
    token_str = Inferno.Tokenizer.decode(tok, [idx-1])
    println("  $(idx-1): ", round(logits[idx], digits=3), " -> '$token_str'")
end

# Check expected tokens
println("\nExpected tokens:")
println("  ' ' (762): ", round(logits[762+1], digits=3))
println("  ' quick' (3842): ", round(logits[3842+1], digits=3))
println("  ' brown' (13478): ", round(logits[13478+1], digits=3))
