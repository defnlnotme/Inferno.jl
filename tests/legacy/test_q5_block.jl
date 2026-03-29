using Inferno
using Statistics

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Get Layer 1 gate weight
gate_q = model_q.layers[2].mlp.gate_weight
gate_f = model_f.layers[2].mlp.gate_weight

println("Quantized gate type: ", typeof(gate_q))
println("Float gate shape: ", size(gate_f))

# Dequantize
gate_dequant = Inferno.QuantsCPU.dequantize_to_array(gate_q)
println("Dequant shape: ", size(gate_dequant))

# Compare first column of dequant with first row of float
println("\nFirst column of dequant (should match first row of float after transpose):")
println("  dequant[:, 1]: ", gate_dequant[1:10, 1])
println("  float[1, :]: ", gate_f[1, 1:10])

# The shapes:
# dequant is (1024, 3584) - column i corresponds to output row i
# float is (3584, 1024) - row i corresponds to output row i
# So dequant[:, i] should equal float[i, :]

println("\nChecking dequant[:, 1] vs float[1, :]:")
println("  Max diff: ", maximum(abs.(gate_dequant[:, 1] .- gate_f[1, :])))

println("\nChecking dequant[:, 100] vs float[100, :]:")
println("  Max diff: ", maximum(abs.(gate_dequant[:, 100] .- gate_f[100, :])))

# Wait - are they supposed to match?
# Let's check the gguf storage format vs our loading

# In GGUF: gate_weight is stored as [hidden, intermediate] = [1024, 3584]
# When loaded as Float32, we transpose to [intermediate, hidden] = [3584, 1024] for matmul
# So float[row, :] is the input-hidden weights for output at row

# In our quantized storage:
# We store inner_dim=1024, outer_dim=3584
# But the data is stored as blocks of 256 elements
# Each row of the output (3584 rows) has 1024 elements / 256 per block = 4 blocks

# The question: how is the data laid out in the quantized format?
# Let's check: num_blocks = 14336 = 3584 * 4 = outer_dim * (inner_dim / 256)

# So blocks are organized as: for each output row, we have 4 blocks of 256 input values
# block 0: output row 0, input values 0-255
# block 1: output row 0, input values 256-511
# etc.

# Let's verify by checking block structure
println("\n=== Block structure ===")
println("num_blocks: ", gate_q.num_blocks)
println("Expected: ", gate_q.outer_dim * (gate_q.inner_dim ÷ 256))

# For row 0, the blocks are at indices 0-3 (0-indexed)
# Block 0 corresponds to input values 0-255 for output row 0

# Dequantize block 0 manually and compare
block_data = gate_q.data[1:176]  # First block (176 bytes for Q5_K)
block_values = Inferno.QuantsCPU.dequantize_q5_k_block(gate_q.data, 1)

println("\nFirst block values [1:10]: ", collect(block_values)[1:10])
println("Expected (from float): ", gate_f[1, 1:10])
