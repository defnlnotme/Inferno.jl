using Inferno

println("Loading model...")
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("\n=== After load_model_cpu ===")
println("model.layers[7] (layer 6) gate_weight[1, 1:5]: ", round.(model.layers[7].mlp.gate_weight[1, 1:5], digits=5))
println("model.layers[6] (layer 5) gate_weight[1, 1:5]: ", round.(model.layers[6].mlp.gate_weight[1, 1:5], digits=5))
