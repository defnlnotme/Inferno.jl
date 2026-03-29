using Inferno
using Statistics

cpu_model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
gpu_model, _ = load_model("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("CPU final_norm weight mean: ", mean(cpu_model.final_norm.weight))
println("GPU final_norm weight mean: ", mean(Float32.(gpu_model.final_norm.weight)))

println("\nCPU in_norm (layer 1) weight mean: ", mean(cpu_model.layers[1].in_norm.weight))
println("GPU in_norm (layer 1) weight mean: ", mean(Float32.(gpu_model.layers[1].in_norm.weight)))
