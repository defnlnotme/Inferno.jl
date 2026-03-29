using Inferno

# Load GGUF file directly
using Inferno.GGUF
file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check embedding tensor dimensions
println("\nTensor shapes:")
for (name, tensor) in file.tensors
    if contains(name, "embed") || contains(name, "token")
        println("  $name: dims=$(tensor.dimensions), type=$(tensor.type)")
    end
    if contains(name, "output") || contains(name, "lm_head")
        println("  $name: dims=$(tensor.dimensions), type=$(tensor.type)")
    end
end

# Check the raw tensor data
println("\nEmbedding tensor details:")
if haskey(file.tensors, "token_embd.weight")
    t = file.tensors["token_embd.weight"]
    println("  token_embd.weight:")
    println("    type: ", t.type)
    println("    dimensions: ", t.dimensions)
    println("    total elements: ", prod(t.dimensions))
end

if haskey(file.tensors, "output.weight")
    t = file.tensors["output.weight"]
    println("  output.weight:")
    println("    type: ", t.type)
    println("    dimensions: ", t.dimensions)
    println("    total elements: ", prod(t.dimensions))
end

# Check metadata
println("\nMetadata:")
for (k, v) in file.metadata
    if contains(k, "embedding") || contains(k, "vocab") || contains(k, "hidden")
        println("  $k = $v")
    end
end
