using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset states
for layer in model.layers
    if layer.is_ssm
        Inferno.ModelCPU.reset_states_cpu!(layer.op)
    end
end

# Test with "The quick"
tokens = Inferno.Tokenizer.encode(tok, "The quick")
println("Tokens: ", tokens)
println("Decoded: ", [Inferno.Tokenizer.decode(tok, [t]) for t in tokens])

# Initialize KV caches
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

# Process first token
logits1 = Inferno.ModelCPU.forward_cpu!(model, [tokens[1]], 0, caches)
println("\nAfter 'The' (token ", tokens[1], "):")
println("  Logits shape: ", size(logits1))
println("  Top 5: ", sortperm(vec(logits1), rev=true)[1:5])

# Process second token
logits2 = Inferno.ModelCPU.forward_cpu!(model, [tokens[2]], 1, caches)
println("\nAfter ' quick' (token ", tokens[2], "):")
println("  Logits shape: ", size(logits2))
println("  Top 5: ", sortperm(vec(logits2), rev=true)[1:5])

# Top tokens for second position
top_k = 10
top_indices = sortperm(vec(logits2), rev=true)[1:top_k]
println("\nTop $top_k after 'The quick':")
for idx in top_indices
    token_str = Inferno.Tokenizer.decode(tok, [idx])
    println("  $idx: ", round(logits2[idx], digits=3), " -> '$token_str'")
end
