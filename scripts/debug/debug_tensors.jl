using Inferno
using Inferno.GGUF

file = read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("Layer 0 tensors (SSM layer):")
for name in sort(collect(keys(file.tensors)))
    if startswith(name, "blk.0.")
        println("  $name: $(file.tensors[name].dimensions)")
    end
end

println("\nLayer 3 tensors (Attention layer):")
for name in sort(collect(keys(file.tensors)))
    if startswith(name, "blk.3.")
        println("  $name: $(file.tensors[name].dimensions)")
    end
end

println("\nToken embedding:")
println("  token_embd.weight: $(file.tensors["token_embd.weight"].dimensions)")

println("\nOutput:")
println("  output.weight: $(file.tensors["output.weight"].dimensions)")
