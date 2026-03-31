using Inferno
using Inferno.GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check all tensor offsets
println("=== All Tensor Offsets ===")
sorted_tensors = sort(collect(file.tensors), by=x->x[2].offset)

for (name, info) in sorted_tensors[1:10]
    println("$name: offset=$(info.offset), dims=$(info.dimensions), type=$(info.type)")
end

# Check what comes before the embedding tensor
println("\n=== Tensors near embedding ===")
for (name, info) in sorted_tensors
    if info.offset < 50000
        println("$name: offset=$(info.offset)")
    end
end

# Check if there's header data before first tensor
first_offset = minimum([info.offset for (name, info) in file.tensors])
println("\nFirst tensor offset: ", first_offset)
println("tensor_data starts at byte: ", 1)

# Check if there's metadata at the beginning of tensor_data
println("\nFirst 100 bytes of tensor_data:")
println(file.tensor_data[1:100])
