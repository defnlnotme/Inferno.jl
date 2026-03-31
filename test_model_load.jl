using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("=== Model Loaded ===")
println("Embedding shape: ", size(model.embed))
println("Embedding norm: ", round(norm(model.embed), digits=5))

# Check token 761 ("The")
println("\nToken 761 embedding:")
println("  norm: ", round(norm(model.embed[:, 762]), digits=5))
println("  first 5: ", round.(model.embed[1:5, 762], digits=5))

# Check lm_head
println("\nlm_head shape: ", size(model.lm_head))
println("lm_head norm: ", round(norm(model.lm_head), digits=5))

# Check if tied
println("\nTied weights check:")
println("  embed' norm: ", round(norm(model.embed'), digits=5))
println("  lm_head == embed': ", model.lm_head ≈ model.embed')
