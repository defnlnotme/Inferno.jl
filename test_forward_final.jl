using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Test with "The"
tokens = Inferno.Tokenizer.encode(tok, "The")
println("Token: ", tokens[1])

# Initialize KV caches
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

# Run forward pass
logits = Inferno.ModelCPU.forward_cpu!(model, tokens, 0, caches)

# Top tokens
top_k = 10
top_indices = sortperm(vec(logits), rev=true)[1:top_k]
println("\nTop $top_k tokens:")
for idx in top_indices
    token_str = Inferno.Tokenizer.decode(tok, [idx-1])  # idx-1 because decode expects 0-indexed
    println("  $(idx-1): ", round(logits[idx], digits=3), " -> '$token_str'")
end
