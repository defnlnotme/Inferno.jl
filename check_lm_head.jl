using Inferno
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check LM head shape
println("LM head shape: ", size(model.lm_head))

# Check a sample row (for token 272)
println("\nLM head row for token 272:")
println("  Shape: ", size(model.lm_head[273, :]))  # 272+1 for 1-indexing
println("  Sample: ", model.lm_head[273, 1:5])

# Check embedding for token 272
println("\nEmbedding for token 272:")
println("  Shape: ", size(model.embed[:, 273]))
println("  Sample: ", model.embed[1:5, 273])

# Compare with LM head
println("\nAre LM head and embedding tied? ")
println("  LM head norm: ", sqrt(sum(abs2.(model.lm_head[273, :]))))
println("  Embedding norm: ", sqrt(sum(abs2.(model.embed[:, 273]))))
