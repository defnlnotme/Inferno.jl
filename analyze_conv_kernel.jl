using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check conv1d for layer 1
layer1 = model.layers[1]
ssm = layer1.op

println("=== Conv1d Kernel Analysis ===")
println("Shape: ", size(ssm.ssm_conv1d))

# For each kernel position, compute average absolute weight across all channels
println("\nAverage |weight| per kernel position:")
for k in 1:ssm.conv_kernel
    avg_abs = sum(abs.(ssm.ssm_conv1d[k, :])) / ssm.conv_channels
    max_abs = maximum(abs.(ssm.ssm_conv1d[k, :]))
    println("  k=$k: avg=$avg_abs, max=$max_abs")
end

# Check if the kernel is causal - newer positions should have larger weights
# A causal conv looks like: y[t] = w[0]*x[t-3] + w[1]*x[t-2] + w[2]*x[t-1] + w[3]*x[t]
# So w[3] should be largest (for newest input x[t])

# Check specific channel
println("\nKernel for channel 1:")
println("  ", round.(ssm.ssm_conv1d[:, 1], digits=5))

println("\nKernel for channel 1000:")
println("  ", round.(ssm.ssm_conv1d[:, 1000], digits=5))

# The ring buffer stores: [x[t-3], x[t-2], x[t-1], x[t]]
# And we compute: output = conv_state[:, 1]*kernel[1] + ... + conv_state[:, 4]*kernel[4]
# If kernel[4] is largest, then newest input has most influence - correct!
