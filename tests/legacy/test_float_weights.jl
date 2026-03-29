using Inferno

println("Loading models...")
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Get the float weights for row 100
gate_f = model_f.layers[2].mlp.gate_weight

println("Float gate weight for row 100:")
println("First 50 values: ", gate_f[101, 1:50])
println("\nValues 160-175 (il=10):")
println(gate_f[101, 161:176])

# The values should be small, typically in range [-0.1, 0.1]
# Let's check the min/max
row100 = gate_f[101, :]
println("\nRow 100 stats:")
println("  min: ", minimum(row100))
println("  max: ", maximum(row100))
println("  mean: ", sum(row100) / length(row100))
println("  std: ", sqrt(sum((row100 .- sum(row100)/length(row100)).^2) / length(row100)))
