using Inferno
using Printf

println("Loading model...")
_, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Print metadata
println("\nMetadata:")
for (k, v) in tok.tokenizer_metadata
 if occursin("qwen", lowercase(k)) || occursin("rope", lowercase(k)) || occursin("attention", lowercase(k))
 println(" $k: $v")
 end
end
