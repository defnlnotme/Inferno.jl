using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check embedding norms for different tokens
tokens = Inferno.Tokenizer.encode(tok, "The")
println("Token 'The' = ", tokens[1])

# Check the embedding at different indices
for offset in 0:2
    idx = tokens[1] + offset
    emb = model.embed[:, idx + 1]  # +1 because Julia is 1-indexed
    println("  embed[:,$(idx+1)] norm: ", round(norm(emb), digits=5))
end

# The correct embedding for token 761 should be at column 762 (761+1)
correct_emb = model.embed[:, tokens[1] + 1]
println("\nCorrect embedding (column 762) norm: ", round(norm(correct_emb), digits=5))
println("First 5 values: ", round.(correct_emb[1:5], digits=5))
