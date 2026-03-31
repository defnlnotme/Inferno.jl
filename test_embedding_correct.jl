using Inferno
using Inferno.GGUF
using Inferno.Dequant

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check embedding tensor
emb_info = file.tensors["token_embd.weight"]
println("Embedding tensor:")
println("  offset: ", emb_info.offset)
println("  data_offset: ", file.data_offset)
println("  total offset: ", file.data_offset + emb_info.offset)
println("  type: ", emb_info.type)

# Calculate correct start position
start = Int(file.data_offset + emb_info.offset) + 1
num_elements = Int(prod(emb_info.dimensions))

println("\n  start: ", start)
println("  num_elements: ", num_elements)

# Dequantize
dequantized = Dequant.dequantize_q6_k(@view(file.tensor_data[start:end]), num_elements)

println("\nDequantized embedding:")
println("  size: ", size(dequantized))
println("  norm: ", round(sqrt(sum(dequantized.^2)), digits=5))
println("  first 10: ", round.(dequantized[1:10], digits=5))

# Check for NaN
nan_count = count(isnan, dequantized)
println("  NaN count: ", nan_count)

# Reshape correctly (GGUF uses row-major)
dims = Int.(emb_info.dimensions)
emb = reshape(dequantized, dims[2], dims[1])'

println("\nFinal embedding matrix:")
println("  shape: ", size(emb))
println("  norm: ", round(norm(emb), digits=5))

# Check token 761 ("The")
println("\nToken 761 embedding:")
println("  norm: ", round(norm(emb[:, 762]), digits=5))
println("  first 5: ", round.(emb[1:5, 762], digits=5))
