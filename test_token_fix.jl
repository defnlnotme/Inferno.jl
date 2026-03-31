using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset all SSM states
for layer in model.layers
    if layer.is_ssm
        Inferno.ModelCPU.reset_states_cpu!(layer.op)
    end
end

# Test with "The" token
tokens = Inferno.Tokenizer.encode(tok, "The")
println("Tokens: ", tokens)

# Initialize KV caches
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

# Run forward pass with the fix
logits = Inferno.ModelCPU.forward_cpu!(model, tokens, 0, caches)

println("\nLogits shape: ", size(logits))
println("Logits norm: ", round(norm(logits), digits=5))
println("Max logit: ", round(maximum(logits), digits=5))
println("Min logit: ", round(minimum(logits), digits=5))

# Top tokens
top_k = 10
top_indices = sortperm(vec(logits), rev=true)[1:top_k]
println("\nTop $top_k tokens:")
for idx in top_indices
    token_str = Inferno.Tokenizer.decode(tok, [idx])
    println("  $idx: ", round(logits[idx], digits=3), " -> '$token_str'")
end
