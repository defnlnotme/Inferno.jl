using Inferno
using LinearAlgebra

function check_weight_values()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    mlp6 = model.layers[6].mlp
    
    println("=== Weight Value Analysis ===")
    
    # Check if weights are dequantized correctly
    # Sample some values
    println("\ngate_weight sample values:")
    println("  [1, 1:5]: ", round.(mlp6.gate_weight[1, 1:5], digits=4))
    println("  [100, 1:5]: ", round.(mlp6.gate_weight[100, 1:5], digits=4))
    println("  [1000, 1:5]: ", round.(mlp6.gate_weight[1000, 1:5], digits=4))
    
    println("\nup_weight sample values:")
    println("  [1, 1:5]: ", round.(mlp6.up_weight[1, 1:5], digits=4))
    println("  [100, 1:5]: ", round.(mlp6.up_weight[100, 1:5], digits=4))
    println("  [1000, 1:5]: ", round.(mlp6.up_weight[1000, 1:5], digits=4))
    
    # Check weight norms
    println("\n=== Weight Norm Analysis ===")
    println("gate_weight:")
    println("  overall norm: ", round(sqrt(sum(abs2.(mlp6.gate_weight))), digits=3))
    println("  row norm mean: ", round(sum([sqrt(sum(abs2.(mlp6.gate_weight[i, :]))) for i in 1:size(mlp6.gate_weight, 1)]) / size(mlp6.gate_weight, 1), digits=3))
    println("  col norm mean: ", round(sum([sqrt(sum(abs2.(mlp6.gate_weight[:, j]))) for j in 1:size(mlp6.gate_weight, 2)]) / size(mlp6.gate_weight, 2), digits=3))
    
    println("\nup_weight:")
    println("  overall norm: ", round(sqrt(sum(abs2.(mlp6.up_weight))), digits=3))
    println("  row norm mean: ", round(sum([sqrt(sum(abs2.(mlp6.up_weight[i, :]))) for i in 1:size(mlp6.up_weight, 1)]) / size(mlp6.up_weight, 1), digits=3))
    println("  col norm mean: ", round(sum([sqrt(sum(abs2.(mlp6.up_weight[:, j]))) for j in 1:size(mlp6.up_weight, 2)]) / size(mlp6.up_weight, 2), digits=3))
    
    # Check if weights are scaled differently
    println("\n=== Weight Scaling Check ===")
    # For Q4_K_XL quantization, weights are dequantized to Float32
    # The dequantization might introduce scaling factors
    
    # Let me check if there's a consistent scaling
    gate_norm = sqrt(sum(abs2.(mlp6.gate_weight)))
    up_norm = sqrt(sum(abs2.(mlp6.up_weight)))
    ratio = gate_norm / up_norm
    
    println("gate_weight norm: ", round(gate_norm, digits=3))
    println("up_weight norm: ", round(up_norm, digits=3))
    println("gate/up ratio: ", round(ratio, digits=3))
    
    # Compare with llama.cpp expected ratio
    llama_gate_norm = 35.623
    llama_up_norm = 14.513
    llama_ratio = llama_gate_norm / llama_up_norm
    
    println("\nllama.cpp:")
    println("  gate norm: ", llama_gate_norm)
    println("  up norm: ", llama_up_norm)
    println("  gate/up ratio: ", round(llama_ratio, digits=3))
    
    println("\nRatio comparison:")
    println("  Our gate/up: ", round(ratio, digits=3))
    println("  llama.cpp gate/up: ", round(llama_ratio, digits=3))
end

check_weight_values()
