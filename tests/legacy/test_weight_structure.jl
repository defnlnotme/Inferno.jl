using Inferno
using Statistics

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Check layer 0 vs layer 1 gate weight structure
println("\n=== Layer 0 Gate Weight ===")
mlp0_q = model_q.layers[1].mlp
gate0 = mlp0_q.gate_weight
println("Type: ", typeof(gate0))
println("inner_dim: ", gate0.inner_dim)
println("outer_dim: ", gate0.outer_dim)
println("num_blocks: ", gate0.num_blocks)
println("data length: ", length(gate0.data))

println("\n=== Layer 1 Gate Weight ===")
mlp1_q = model_q.layers[2].mlp
gate1 = mlp1_q.gate_weight
println("Type: ", typeof(gate1))
println("inner_dim: ", gate1.inner_dim)
println("outer_dim: ", gate1.outer_dim)
println("num_blocks: ", gate1.num_blocks)
println("data length: ", length(gate1.data))

# Dequantize both and compare with Float32 versions
println("\n=== Comparing Dequantized Values ===")
gate0_dequant = Inferno.QuantsCPU.dequantize_to_array(gate0)
gate0_f = model_f.layers[1].mlp.gate_weight

# Check first row
println("Layer 0 gate, first row (dequant): ", gate0_dequant[1:5, 1])
println("Layer 0 gate, first row (float): ", gate0_f[1, 1:5])
println("Match: ", maximum(abs.(gate0_dequant[:, 1] .- gate0_f[1, :])) < 1e-5)

# Layer 1
gate1_dequant = Inferno.QuantsCPU.dequantize_to_array(gate1)
gate1_f = model_f.layers[2].mlp.gate_weight

println("\nLayer 1 gate, first row (dequant): ", gate1_dequant[1:5, 1])
println("Layer 1 gate, first row (float): ", gate1_f[1, 1:5])
println("Match: ", maximum(abs.(gate1_dequant[:, 1] .- gate1_f[1, :])) < 1e-5)

# Check if there's something wrong with the indexing
println("\n=== Checking Matrix Layout ===")
println("Dequant shape: ", size(gate1_dequant))
println("Float shape: ", size(gate1_f))

# The issue: maybe the outer/inner dims are swapped for layer 1?
# Let's check: if inner=1024, outer=3584, then dequant should be [1024, 3584]
# But float might be [3584, 1024] (transposed)

# Try transposed comparison
println("\nTrying transposed comparison for Layer 1:")
println("dequant[1, 1:5]: ", gate1_dequant[1, 1:5])
println("float[1:5, 1]: ", gate1_f[1:5, 1])

# What if the dequant is [outer, inner] instead of [inner, outer]?
println("\nWhat if dequant is [outer, inner]?")
println("dequant[1:5, 1]: ", gate1_dequant[1:5, 1])
println("float[1, 1:5]: ", gate1_f[1, 1:5])
