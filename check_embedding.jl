using Inferno
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check embedding for specific tokens
println("=== Embedding check ===")

# Token 761 = "The"
emb_the = model.embed[:, 762]  # +1 for 1-indexing
println("Token 761 (\"The\"):")
println("  Norm: ", sqrt(sum(abs2.(emb_the))))
println("  Sample: ", emb_the[1:5])

# Check if embedding matches what Python sees
# We can compare with the GGUF file directly
file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get tok_embd.weight
tok_embd = file.tensors["tok_embd.weight"]
println("\nGGUF tok_embd.weight:")
println("  Shape: ", size(tok_embd.data))

# Check the same token
println("\nToken 761 from GGUF:")
println("  Sample: ", tok_embd.data[1:5, 762])
