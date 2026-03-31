using Inferno

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check token indexing
prompt = "The"
tokens = Inferno.Tokenizer.encode(tok, prompt)
println("Tokens from encode: ", tokens)
println("Type: ", typeof(tokens))

# Check embedding matrix size
println("\nEmbedding matrix size: ", size(model.embed))
println("Expected: (hidden_size, vocab_size) = (1024, vocab)")

# Check if we're indexing correctly
# If tokens are 1-indexed, we should use them directly
# If tokens are 0-indexed, we need to add 1

# The forward_cpu! uses: x = view(model.embed, :, tok)
# This means it expects tok to be the column index

# Let's test
tok = tokens[1]
println("\nToken value: ", tok)
println("Embedding column $tok norm: ", round(sqrt(sum(model.embed[:, tok].^2)), digits=5))

# Compare with tok+1
println("Embedding column $(tok+1) norm: ", round(sqrt(sum(model.embed[:, tok+1].^2)), digits=5))
