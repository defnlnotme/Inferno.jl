using Inferno
using Statistics
using Random

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Layer 1 = model_q.layers[2] (1-indexed)
# Layer 1 uses Q5_K for gate/up

println("\n=== Testing Layer 1 MLP directly ===")
mlp_q = model_q.layers[2].mlp
mlp_f = model_f.layers[2].mlp

Random.seed!(42)
test_input = randn(Float32, 1024)

println("\nGate weight types: quant=$(typeof(mlp_q.gate_weight)), float=$(typeof(mlp_f.gate_weight))")

# Test gate multiplication
gate_out_q = ModelCPU.mlp_mat_vec_mul(mlp_q.gate_weight, test_input)
gate_out_f = mlp_f.gate_weight * test_input

println("\nGate multiplication:")
println("  Quantized [1:5]: ", gate_out_q[1:5])
println("  Float [1:5]: ", gate_out_f[1:5])
println("  Max diff: ", maximum(abs.(gate_out_q .- gate_out_f)))

# Dequantize and compare
gate_dequant = Inferno.QuantsCPU.dequantize_to_array(mlp_q.gate_weight)
println("\nDequantized gate shape: ", size(gate_dequant))
println("Float gate shape: ", size(mlp_f.gate_weight))

# Try manual multiplication
manual_gate_out = gate_dequant' * test_input
println("\nManual multiplication (dequant' * x):")
println("  Manual [1:5]: ", manual_gate_out[1:5])
println("  Float [1:5]: ", gate_out_f[1:5])
println("  Max diff: ", maximum(abs.(manual_gate_out .- gate_out_f)))
