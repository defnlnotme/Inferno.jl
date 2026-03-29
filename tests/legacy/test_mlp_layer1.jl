using Inferno
using Statistics
using Random

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Test both layer 0 and layer 1 MLPs with same input
Random.seed!(42)
test_input = randn(Float32, 1024)

println("\n=== Layer 0 MLP ===")
mlp0_q = model_q.layers[1].mlp
mlp0_f = model_f.layers[1].mlp
out0_q = ModelCPU.mlp_forward(mlp0_q, test_input)
out0_f = ModelCPU.mlp_forward(mlp0_f, test_input)
println("Quantized output [1:5]: ", out0_q[1:5])
println("Float output [1:5]: ", out0_f[1:5])
println("Max diff: ", maximum(abs.(out0_q .- out0_f)))

println("\n=== Layer 1 MLP ===")
mlp1_q = model_q.layers[2].mlp
mlp1_f = model_f.layers[2].mlp
out1_q = ModelCPU.mlp_forward(mlp1_q, test_input)
out1_f = ModelCPU.mlp_forward(mlp1_f, test_input)
println("Quantized output [1:5]: ", out1_q[1:5])
println("Float output [1:5]: ", out1_f[1:5])
println("Max diff: ", maximum(abs.(out1_q .- out1_f)))

# Check weight dimensions
println("\n=== Weight Dimensions ===")
println("Layer 0 gate: inner=$(mlp0_q.gate_weight.inner_dim), outer=$(mlp0_q.gate_weight.outer_dim)")
println("Layer 1 gate: inner=$(mlp1_q.gate_weight.inner_dim), outer=$(mlp1_q.gate_weight.outer_dim)")

# Compare dequantized weights
println("\n=== Dequantized vs Float weights ===")
gate1_q_dequant = Inferno.QuantsCPU.dequantize_to_array(mlp1_q.gate_weight)
println("Layer 1 gate (quantized, dequantized) [1:5, 1]: ", gate1_q_dequant[1:5, 1])
println("Layer 1 gate (float) [1, 1:5]: ", mlp1_f.gate_weight[1, 1:5])
println("Note: quantized is stored as [inner, outer], float is transposed to [outer, inner]")

# Test single multiplication
println("\n=== Test Layer 1 gate multiplication ===")
x = test_input
out_gate_q = ModelCPU.mlp_mat_vec_mul(mlp1_q.gate_weight, x)
out_gate_f = mlp1_f.gate_weight * x
println("Gate output (quantized) [1:5]: ", out_gate_q[1:5])
println("Gate output (float) [1:5]: ", out_gate_f[1:5])
println("Max diff: ", maximum(abs.(out_gate_q .- out_gate_f)))
