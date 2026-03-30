using Inferno
using LinearAlgebra

function dump_weight_values()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    mlp6 = model.layers[6].mlp
    
    println("=== gate_weight values ===")
    println("Row 1, first 20: ", round.(mlp6.gate_weight[1, 1:20], digits=4))
    println("Row 100, first 20: ", round.(mlp6.gate_weight[100, 1:20], digits=4))
    println("Row 1000, first 20: ", round.(mlp6.gate_weight[1000, 1:20], digits=4))
    
    println("\n=== up_weight values ===")
    println("Row 1, first 20: ", round.(mlp6.up_weight[1, 1:20], digits=4))
    println("Row 100, first 20: ", round.(mlp6.up_weight[100, 1:20], digits=4))
    println("Row 1000, first 20: ", round.(mlp6.up_weight[1000, 1:20], digits=4))
    
    # Check if values have different distributions
    gate_mean = sum(mlp6.gate_weight) / length(mlp6.gate_weight)
    up_mean = sum(mlp6.up_weight) / length(mlp6.up_weight)
    
    gate_std = sqrt(sum((mlp6.gate_weight .- gate_mean).^2) / length(mlp6.gate_weight))
    up_std = sqrt(sum((mlp6.up_weight .- up_mean).^2) / length(mlp6.up_weight))
    
    println("\n=== Statistics ===")
    println("gate_weight: mean=$(round(gate_mean, digits=5)), std=$(round(gate_std, digits=4))")
    println("up_weight: mean=$(round(up_mean, digits=5)), std=$(round(up_std, digits=4))")
    
    # Check element-wise ratio
    # Skip zeros
    valid_idx = findall(x -> abs(x) > 1e-6, mlp6.up_weight)
    ratio_vals = mlp6.gate_weight[valid_idx] ./ mlp6.up_weight[valid_idx]
    
    println("\n=== Element-wise ratio (gate / up) ===")
    println("Mean: ", round(sum(ratio_vals) / length(ratio_vals), digits=3))
    println("Std: ", round(sqrt(sum((ratio_vals .- sum(ratio_vals)/length(ratio_vals)).^2) / length(ratio_vals)), digits=3))
    println("Min: ", round(minimum(ratio_vals), digits=3))
    println("Max: ", round(maximum(ratio_vals), digits=3))
end

dump_weight_values()
