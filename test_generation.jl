using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Generate multiple tokens using the generate_text function
prompt = "The quick brown fox"
result = Inferno.Generate.generate_text(model, tok, prompt; max_tokens=30, temperature=0.1f0)
println("\n=== Generated Text ===")
println(result)
