using Inferno
using Statistics

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

gate_q = model_q.layers[2].mlp.gate_weight
gate_f = model_f.layers[2].mlp.gate_weight

gate_dequant = Inferno.QuantsCPU.dequantize_to_array(gate_q)

# Find the elements with largest differences for row 100
row = 100
diffs = abs.(gate_dequant[:, row+1] .- gate_f[row+1, :])

# Find indices of largest differences
sorted_indices = sortperm(diffs, rev=true)

println("\nElements with largest differences for row $row:")
for i in 1:10
 idx = sorted_indices[i]
 println("  idx=$idx: dequant=$(gate_dequant[idx, row+1]), float=$(gate_f[row+1, idx]), diff=$(diffs[idx])")
end

# Check if the pattern suggests something about block boundaries
# Each block has 256 elements
# Elements 0-255, 256-511, 512-767, 768-1023
println("\nBlock boundary analysis:")
for block in 0:3
 start = block * 256 + 1
 stop = min((block + 1) * 256, 1024)
 block_diffs = diffs[start:stop]
 println("  Block $block (indices $start-$stop): max=$(maximum(block_diffs)), mean=$(mean(block_diffs))")
end

# Let me check if block 401-403 are correct
println("\n=== Checking each block for row $row ===")
for b in 0:3
 block_idx = 4 * row + b
 block_offset = block_idx * 176 + 1
 block_values = Inferno.QuantsCPU.dequantize_q5_k_block(gate_q.data, block_offset)
 
 # These values should go into positions b*256 to (b+1)*256-1
 start_idx = b * 256 + 1
 end_idx = (b + 1) * 256
 
 # Compare with expected from float
 expected = gate_f[row+1, start_idx:end_idx]
 actual = collect(block_values)
 
 diff = maximum(abs.(actual .- expected))
 println("Block $block_idx (input indices $(start_idx)-$(end_idx)): max_diff=$diff")
 if diff > 0.01
 println("  First 5 actual: ", actual[1:5])
 println("  First 5 expected: ", expected[1:5])
 end
end
