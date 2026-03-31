using Inferno
using Inferno.GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get conv1d weight info
conv_info = file.tensors["blk.0.ssm_conv1d.weight"]
println("Conv1d tensor:")
println("  dimensions: ", conv_info.dimensions)
println("  type: ", conv_info.type)

# The GGUF stores as (d_conv, d_inner) = (4, 6144)
# But when we read it with extract_tensor_cpu, what do we get?

# Check if the tensor is stored row-major or column-major
# In GGUF, tensors are stored in row-major (C-style) order
# When we reshape in Julia, we get column-major (Julia-style) order

# Let's manually read the data
start = Int(file.data_offset + conv_info.offset) + 1
dims = Int.(conv_info.dimensions)  # [4, 6144]

println("\nExpected dimensions: $dims")
println("Total elements: ", prod(dims))

# Read as Float32
data = reinterpret(Float32, @view file.tensor_data[start:start+prod(dims)*4-1])
println("\nRaw data size: ", size(data))

# The data is stored row-major as [4, 6144]
# In row-major: data[0] is row 0, col 0; data[1] is row 0, col 1; ...
# To convert to Julia column-major, we need to reshape and transpose

# Wrong interpretation (just reshape)
wrong = reshape(data, dims[1], dims[2])
println("\nWrong reshape: ", size(wrong))
println("  row 1 norm: ", round(sum(abs.(wrong[1, :])), digits=5))
println("  row 4 norm: ", round(sum(abs.(wrong[4, :])), digits=5))

# Correct interpretation (reshape then transpose)
correct = reshape(data, dims[2], dims[1])'
println("\nCorrect reshape+transpose: ", size(correct))
println("  row 1 norm: ", round(sum(abs.(correct[1, :])), digits=5))
println("  row 4 norm: ", round(sum(abs.(correct[4, :])), digits=5))

# Check if they match
println("\nwrong[1,1:5]: ", round.(wrong[1, 1:5], digits=5))
println("correct[1,1:5]: ", round.(correct[1, 1:5], digits=5))
