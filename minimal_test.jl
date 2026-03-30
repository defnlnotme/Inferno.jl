using Inferno
using Inferno.GGUF
using Inferno.Dequant

# Load GGUF file
file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get tensor info
gate_info = file.tensors["blk.6.ffn_gate.weight"]
println("Gate tensor info:")
println("  dimensions: ", gate_info.dimensions)
println("  type: ", gate_info.type)

# Dequantize
start = Int(file.data_offset + gate_info.offset) + 1
num_elements = Int(prod(gate_info.dimensions))
data = Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)

println("\nDequantized data:")
println("  length: ", length(data))
println("  data[1:5]: ", round.(data[1:5], digits=5))

# Reshape with correct fix
dims = Tuple(Int.(gate_info.dimensions))
inner = dims[1]
outer = dims[2]

println("\nDimensions:")
println("  inner = ", inner)
println("  outer = ", outer)

# Apply reshape fix
M = reshape(data, outer, inner)'
println("\nAfter reshape(data, outer, inner)':")
println("  shape: ", size(M))
println("  M[1, 1:5]: ", round.(M[1, 1:5], digits=5))

# For FFN, we need (intermediate, hidden) = (3584, 1024)
# After transpose: shape is (1024, 3584)
# We need to transpose again: shape is (3584, 1024)
gate_weight = Matrix(Float32.(M'))
println("\ngate_weight (for FFN):")
println("  shape: ", size(gate_weight))
println("  gate_weight[1, 1:5]: ", round.(gate_weight[1, 1:5], digits=5))
