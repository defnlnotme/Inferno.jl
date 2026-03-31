using Inferno
using Inferno.GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("=== Checking for output.weight ===")
if haskey(file.tensors, "output.weight")
    info = file.tensors["output.weight"]
    println("output.weight found:")
    println("  dimensions: ", info.dimensions)
    println("  type: ", info.type)
else
    println("output.weight NOT found - using tied embedding")
end

# Check embedding tensor
println("\n=== Embedding tensor ===")
emb_info = file.tensors["token_embd.weight"]
println("token_embd.weight:")
println("  dimensions: ", emb_info.dimensions)
println("  type: ", emb_info.type)
