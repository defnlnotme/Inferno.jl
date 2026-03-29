using Inferno
using Statistics

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

gate_q = model_q.layers[2].mlp.gate_weight
gate_f = model_f.layers[2].mlp.gate_weight

# Dequantize the full weight matrix
gate_dequant = Inferno.QuantsCPU.dequantize_to_array(gate_q)

# Check the layout
# Q5_K has 176 bytes per block
# We have 14336 blocks for a (1024, 3584) matrix
# Layout: for each output row (3584), we have inner_dim/256 = 4 blocks

# So block indices for output row r (0-indexed) are: 4*r, 4*r+1, 4*r+2, 4*r+3
# These blocks give us input values 0-255, 256-511, 512-767, 768-1023 for that row

println("Checking data layout...")

# Check output row 0
println("\n=== Output Row 0 ===")
println("Blocks: 0, 1, 2, 3")
println("Dequant values for row 0 (first 10): ", gate_dequant[1:10, 1])
println("Float values for row 0 (first 10): ", gate_f[1, 1:10])

# Check output row 100
row = 100
println("\n=== Output Row $row ===")
# Blocks for row 100: 4*100=400, 401, 402, 403
block_start = 4 * row
println("Blocks: $block_start, $(block_start+1), $(block_start+2), $(block_start+3)")

# Dequant should have row 100 at index [1:1024, 101]
println("Dequant[:, $(row+1)] first 10: ", gate_dequant[1:10, row+1])
println("Float[$(row+1), :] first 10: ", gate_f[row+1, 1:10])

# Let me manually dequant block 400 and check
block_400_values = Inferno.QuantsCPU.dequantize_q5_k_block(gate_q.data, 400 * 176 + 1)
println("\nBlock 400 values (first 10): ", collect(block_400_values)[1:10])

# This should correspond to input indices 0-255 for output row 100
# So dequant[1:256, 101] should equal block_400_values
println("\nDequant[1:10, $(row+1)]: ", gate_dequant[1:10, row+1])

# Check full column comparison
max_diff_row100 = maximum(abs.(gate_dequant[:, row+1] .- gate_f[row+1, :]))
println("\nMax diff for row $row: $max_diff_row100")

# Let's check the percentage of elements that match within tolerance
matches = sum(abs.(gate_dequant[:, row+1] .- gate_f[row+1, :]) .< 0.01)
println("Elements matching within 0.01: $matches / $(length(gate_dequant[:, row+1]))")
