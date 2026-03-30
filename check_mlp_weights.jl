using Inferno
using LinearAlgebra

function check_mlp_weights()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    mlp6 = model.layers[6].mlp
    
    println("=== Layer 6 MLP Weights ===")
    println("\ngate_weight (shape: ", size(mlp6.gate_weight), "):")
    println("  norm: ", round(sqrt(sum(abs2.(mlp6.gate_weight))), digits=3))
    println("  min: ", round(minimum(mlp6.gate_weight), digits=3))
    println("  max: ", round(maximum(mlp6.gate_weight), digits=3))
    println("  mean: ", round(sum(mlp6.gate_weight) / length(mlp6.gate_weight), digits=4))
    println("  sample [1:3, 1:3]:")
    for i in 1:3
        println("    ", round.(mlp6.gate_weight[i, 1:3], digits=4))
    end
    
    println("\nup_weight (shape: ", size(mlp6.up_weight), "):")
    println("  norm: ", round(sqrt(sum(abs2.(mlp6.up_weight))), digits=3))
    println("  min: ", round(minimum(mlp6.up_weight), digits=3))
    println("  max: ", round(maximum(mlp6.up_weight), digits=3))
    println("  mean: ", round(sum(mlp6.up_weight) / length(mlp6.up_weight), digits=4))
    println("  sample [1:3, 1:3]:")
    for i in 1:3
        println("    ", round.(mlp6.up_weight[i, 1:3], digits=4))
    end
    
    println("\ndown_weight (shape: ", size(mlp6.down_weight), "):")
    println("  norm: ", round(sqrt(sum(abs2.(mlp6.down_weight))), digits=3))
    println("  min: ", round(minimum(mlp6.down_weight), digits=3))
    println("  max: ", round(maximum(mlp6.down_weight), digits=3))
    println("  mean: ", round(sum(mlp6.down_weight) / length(mlp6.down_weight), digits=4))
    println("  sample [1:3, 1:3]:")
    for i in 1:3
        println("    ", round.(mlp6.down_weight[i, 1:3], digits=4))
    end
    
    # Check if weights are dequantized correctly
    # For Q4_K_XL quantization, weights should be dequantized to Float32
    println("\n=== Weight Quantization Check ===")
    println("Weight type: ", typeof(mlp6.gate_weight))
    println("Weight eltype: ", eltype(mlp6.gate_weight))
    
    # Check if there's any pattern in the weights
    println("\n=== Weight Pattern Analysis ===")
    gate_row_norms = [sqrt(sum(abs2.(mlp6.gate_weight[i, :]))) for i in 1:size(mlp6.gate_weight, 1)]
    up_row_norms = [sqrt(sum(abs2.(mlp6.up_weight[i, :]))) for i in 1:size(mlp6.up_weight, 1)]
    down_row_norms = [sqrt(sum(abs2.(mlp6.down_weight[i, :]))) for i in 1:size(mlp6.down_weight, 1)]
    
    println("gate_weight row norms: min=$(round(minimum(gate_row_norms), digits=3)), max=$(round(maximum(gate_row_norms), digits=3)), mean=$(round(sum(gate_row_norms)/length(gate_row_norms), digits=3))")
    println("up_weight row norms: min=$(round(minimum(up_row_norms), digits=3)), max=$(round(maximum(up_row_norms), digits=3)), mean=$(round(sum(up_row_norms)/length(up_row_norms), digits=3))")
    println("down_weight row norms: min=$(round(minimum(down_row_norms), digits=3)), max=$(round(maximum(down_row_norms), digits=3)), mean=$(round(sum(down_row_norms)/length(down_row_norms), digits=3))")
end

check_mlp_weights()
