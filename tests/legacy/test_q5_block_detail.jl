using Inferno

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

gate_q = model_q.layers[2].mlp.gate_weight
gate_f = model_f.layers[2].mlp.gate_weight

# Dequantize block 400 and check every element
block_idx = 400
block_offset = block_idx * 176 + 1
block_values = Inferno.QuantsCPU.dequantize_q5_k_block(gate_q.data, block_offset)

# Get the expected values
expected = gate_f[101, 1:256]

println("Block 400 detailed comparison (first 50 elements):")
println("Index | Actual | Expected | Diff")
println("-" ^ 50)
for i in 1:50
 actual = collect(block_values)[i]
 exp = expected[i]
 diff = abs(actual - exp)
 println("$i | $actual | $exp | $diff")
end

# Check positions with largest errors in block
errors = abs.(collect(block_values) .- expected)
sorted_errors = sortperm(errors, rev=true)

println("\nPositions with largest errors in block 400:")
for i in 1:20
 idx = sorted_errors[i]
 println("idx=$idx: actual=$(collect(block_values)[idx]), expected=$(expected[idx]), error=$(errors[idx])")
end
