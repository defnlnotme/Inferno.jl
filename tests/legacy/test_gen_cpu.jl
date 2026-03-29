using Inferno
using Printf
using Inferno.Generate: chat

println("Loading model...")
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

messages = [
 ("user", "What is the capital of France?")
]

println("\nGenerating...")
output = chat(model, tok, messages; max_tokens=50, temperature=0.0f0)
println("\nResponse: $output")
