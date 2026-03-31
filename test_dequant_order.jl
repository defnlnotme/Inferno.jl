using Inferno
using Inferno.GGUF
using Inferno.Dequant

# Read the GGUF file
file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get the embedding tensor info
emb_info = file.tensors["token_embd.weight"]
println("Embedding tensor:")
println("  dimensions: ", emb_info.dimensions)
println("  type: ", emb_info.type)

# Get the raw data offset and size
num_elements = Int(prod(emb_info.dimensions))
dims = Int.(emb_info.dimensions)

# Dequantize
start_int = Int(emb_info.offset)
raw_data = @view file.tensor_data[start_int:end]
dequantized = Dequant.dequantize_q6_k(raw_data, num_elements)

println("\nDequantized embedding:")
println("  size: ", size(dequantized))
println("  norm: ", round(sqrt(sum(dequantized.^2)), digits=5))
println("  first 10: ", round.(dequantized[1:10], digits=5))

# Reshape to matrix
# dimensions are [hidden, vocab] in GGUF C-style
# We need to reshape correctly
inner = dims[1]
outer = dims[2]

# The dequantized data is in row-major order (C-style)
# We need to interpret it correctly for Julia's column-major order
# reshape(dequantized, inner, outer) assumes column-major
# reshape(dequantized, outer, inner)' transposes to get correct interpretation

emb_wrong = reshape(dequantized, inner, outer)
emb_correct = reshape(dequantized, outer, inner)'

println("\nEmb wrong shape: ", size(emb_wrong))
println("Emb correct shape: ", size(emb_correct))

# Check "The" token (761)
println("\nToken 761 embedding:")
println("  emb_wrong[:, 762] norm: ", round(sqrt(sum(emb_wrong[:, 762].^2)), digits=5))
println("  emb_correct[:, 762] norm: ", round(sqrt(sum(emb_correct[:, 762].^2)), digits=5))
