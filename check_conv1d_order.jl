using Inferno
using Inferno.GGUF
using Inferno.Dequant

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get conv1d weight info
conv_info = file.tensors["blk.0.ssm_conv1d.weight"]
println("Conv1d tensor:")
println("  dimensions: ", conv_info.dimensions)
println("  type: ", conv_info.type)

# Load the raw data
start = Int(file.data_offset + conv_info.offset) + 1
data = @view file.tensor_data[start:start+100]

# For F32, each value is 4 bytes
println("\nFirst few floats:")
for i in 1:4
    val = reinterpret(Float32, data[(i-1)*4+1:i*4])[1]
    println("  Position $i: ", round(val, digits=5))
end

# Check if the kernel is stored in reverse order
# In PyTorch, Conv1d weight is (out_channels, in_channels, kernel_size)
# But for causal conv1d without input channel dimension, it might be (out_channels, kernel_size)
# GGUF might store it as (kernel_size, out_channels) or (out_channels, kernel_size)

println("\n=== Checking Dimensions ===")
println("Expected: (kernel=4, channels=6144) = 4 * 6144 = 24576 elements")
println("Actual dimensions: ", conv_info.dimensions)
println("Actual total: ", prod(conv_info.dimensions))

# The conv1d should be applied as: output[c] = sum_k(input[c,k] * weight[k,c])
# For a causal conv, weight[0,c] multiplies the oldest input, weight[3,c] multiplies newest
# If GGUF stores weight in reverse, we need to flip along kernel dimension
