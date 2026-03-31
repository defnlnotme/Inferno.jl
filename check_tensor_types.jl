using Inferno
using Inferno.GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check tensor types
println("=== Tensor Types ===")
for (name, info) in file.tensors
    if occursin("embd", name) || occursin("output", name) || occursin("blk.0", name)
        println("$name: $(info.type)")
    end
end

# Check embedding type specifically
emb_info = file.tensors["token_embd.weight"]
println("\n=== Embedding ===")
println("Type: ", emb_info.type)
println("GGML_TYPE_Q6_K = ", GGUF.GGML_TYPE_Q6_K)

# Check actual tensor type enum value
println("Enum value: ", Int(emb_info.type))
