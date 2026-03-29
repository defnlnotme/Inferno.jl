using Inferno
using Random

# Load the same model twice
println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

mlp_q = model_q.layers[1].mlp
mlp_f = model_f.layers[1].mlp

# Create a random input vector
Random.seed!(42)
x = randn(Float32, 1024)

# Test multiplication
println("\nTesting matrix-vector multiplication...")

# Quantized path
out_q = ModelCPU.mlp_mat_vec_mul(mlp_q.gate_weight, x)
println("  Quantized output size: ", size(out_q))
println("  Quantized output [1:5]: ", out_q[1:5])

# Float path
out_f = mlp_f.gate_weight * x
println("  Float output size: ", size(out_f))
println("  Float output [1:5]: ", out_f[1:5])

# Compare
diff = out_q .- out_f
println("\n  Max diff: ", maximum(abs.(diff)))
println("  Mean diff: ", sum(abs.(diff)) / length(diff))
println("  Relative error: ", sum(abs.(diff)) / sum(abs.(out_f)) * 100, " %")
