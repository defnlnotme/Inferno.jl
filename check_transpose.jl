using Inferno

file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Call extract_tensor_cpu
gate_tensor = Inferno.LoaderCPU.extract_tensor_cpu(file, "blk.6.ffn_gate.weight")

println("gate_tensor type: ", typeof(gate_tensor))
println("gate_tensor shape: ", size(gate_tensor))
println("gate_tensor[1, 1:5]: ", round.(gate_tensor[1, 1:5], digits=5))

# After transpose (as in load_mlp)
gate_weight = Matrix(Float32.(gate_tensor'))

println("\nafter Matrix(Float32.(gate_tensor)'):")
println("gate_weight shape: ", size(gate_weight))
println("gate_weight[1, 1:5]: ", round.(gate_weight[1, 1:5], digits=5))

# Load actual model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
mlp6 = model.layers[6].mlp

println("\nActual model gate_weight:")
println("gate_weight shape: ", size(mlp6.gate_weight))
println("gate_weight[1, 1:5]: ", round.(mlp6.gate_weight[1, 1:5], digits=5))
