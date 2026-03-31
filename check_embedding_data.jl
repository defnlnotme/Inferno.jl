using Inferno
using Inferno.GGUF
using Inferno.Dequant

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check embedding tensor
emb_info = file.tensors["token_embd.weight"]
println("Embedding tensor:")
println("  offset: ", emb_info.offset)
println("  dimensions: ", emb_info.dimensions)
println("  type: ", emb_info.type)

# Calculate expected data size
num_elements = Int(prod(emb_info.dimensions))
num_blocks = num_elements ÷ 256
block_size = 210
expected_bytes = num_blocks * block_size

println("\n  num_elements: ", num_elements)
println("  num_blocks: ", num_blocks)
println("  expected_bytes: ", expected_bytes)
println("  actual data size: ", length(file.tensor_data))

# Check if offset + expected_bytes is within bounds
end_offset = Int(emb_info.offset) + expected_bytes
println("  end_offset: ", end_offset)
println("  within bounds: ", end_offset <= length(file.tensor_data))

# Check the first few bytes at the offset
offset = Int(emb_info.offset)
println("\nFirst 20 bytes at offset:")
println("  ", file.tensor_data[offset:offset+19])

# Try dequantizing just one block
one_block = @view file.tensor_data[offset:offset+block_size-1]
println("\nOne block data:")
println("  first 20 bytes: ", one_block[1:20])

# Manual dequantization of first block
d = Float32(reinterpret(Float16, one_block[209:210])[1])
println("\nScale d: ", d)
