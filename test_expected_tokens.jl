using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Test with "The"
tokens = Inferno.Tokenizer.encode(tok, "The")
println("Token ID for 'The': ", tokens[1])

# Check what the correct next token should be
# For "The", the model should predict " " (space) or " quick" etc.

# Let's check the logits for common tokens
common_tokens = [
    (" ", 762),
    (" quick", 3842),
    (" brown", 15757),
    (" first", 1058),
    (" most", 1584),
]

println("\nLogits for expected next tokens:")
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
logits = Inferno.ModelCPU.forward_cpu!(model, tokens, 0, caches)

for (name, tok_id) in common_tokens
    # tok_id is 0-indexed, logits are 1-indexed
    logit_val = logits[tok_id + 1]
    println("  '$name' ($tok_id): ", round(logit_val, digits=3))
end

# Also check the actual top token
top_idx = argmax(vec(logits))
println("\nTop token index: $top_idx (0-indexed: $(top_idx-1))")
println("Top token: '", Inferno.Tokenizer.decode(tok, [top_idx-1]), "'")
