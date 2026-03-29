using Inferno

# Load the same model twice - once quantized, once dequantized
println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Get first layer MLP
mlp_q = model_q.layers[1].mlp
mlp_f = model_f.layers[1].mlp

println("\nWeight dimensions:")
println("  Quantized gate: inner=$(mlp_q.gate_weight.inner_dim), outer=$(mlp_q.gate_weight.outer_dim)")
println("  Float gate: ", size(mlp_f.gate_weight))

# Test: Dequantize the quantized weight and compare
println("\nDequantizing quantized gate weight...")
gate_dequant = Inferno.QuantsCPU.dequantize_to_array(mlp_q.gate_weight)
println("  Dequantized shape: ", size(gate_dequant))

# Compare with Float32 weight (which was transposed during loading)
# gate_f is [outer, inner] = [3584, 1024] after transpose
# gate_dequant is [inner, outer] = [1024, 3584] (no transpose)

println("\nComparing values...")
println("  Float gate [1:5, 1]: ", mlp_f.gate_weight[1:5, 1])
println("  Dequant [1:5, 1]: ", gate_dequant[1:5, 1])

# The issue: Float weight was transposed, so gate_f[row, col] = dequant[col, row]
println("\n  Float gate[1, 1:5]: ", mlp_f.gate_weight[1, 1:5])
println("  Dequant[1:5, 1]: ", gate_dequant[1:5, 1])

# Check if they match after accounting for transpose
println("\n  Float gate[1, 1:5] should equal Dequant[1:5, 1]:")
println("  Float gate[1, 1:5] = ", mlp_f.gate_weight[1, 1:5])
println("  Dequant[1:5, 1] = ", gate_dequant[1:5, 1])

# Compute difference
diff = mlp_f.gate_weight[1, 1:5] .- gate_dequant[1:5, 1]
println("  Difference: ", diff)
