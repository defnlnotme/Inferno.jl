using Inferno
using LinearAlgebra
using GGUF

function investigate_dequantization()
    # Q4_K_XL is a quantization format
    # Let me check how we're loading the weights
    
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    mlp6 = model.layers[6].mlp
    
    println("=== Q4_K_XL Dequantization Investigation ===")
    
    # Check the weight structure
    println("\nWeight types:")
    println("  gate_weight: ", typeof(mlp6.gate_weight))
    println("  up_weight: ", typeof(mlp6.up_weight))
    
    # Let me check if the weights have unusual patterns
    println("\n=== Weight Statistics ===")
    
    for (name, weight) in [("gate", mlp6.gate_weight), ("up", mlp6.up_weight)]
        println("\n$name weight:")
        println("  shape: ", size(weight))
        println("  min: ", round(minimum(weight), digits=4))
        println("  max: ", round(maximum(weight), digits=4))
        println("  mean: ", round(sum(weight) / length(weight), digits=6))
        println("  std: ", round(sqrt(sum((weight .- sum(weight)/length(weight)).^2) / length(weight)), digits=4))
        
        # Check for zero rows/cols
        row_norms = [sqrt(sum(abs2.(weight[i, :]))) for i in 1:size(weight, 1)]
        zero_rows = count(x -> x < 1e-6, row_norms)
        println("  zero rows: $zero_rows / $(size(weight, 1))")
        
        col_norms = [sqrt(sum(abs2.(weight[:, j]))) for j in 1:size(weight, 2)]
        zero_cols = count(x -> x < 1e-6, col_norms)
        println("  zero cols: $zero_cols / $(size(weight, 2))")
    end
    
    # Check the actual tensor info from GGUF
    println("\n=== GGUF Tensor Info ===")
    file = GGUF.GGUFFile("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    up_info = file.tensors["blk.6.ffn_up.weight"]
    
    println("gate tensor:")
    println("  name: ", gate_info.name)
    println("  dimensions: ", gate_info.dimensions)
    println("  type: ", gate_info.type)
    
    println("\nup tensor:")
    println("  name: ", up_info.name)
    println("  dimensions: ", up_info.dimensions)
    println("  type: ", up_info.type)
    
    # Check if the quantization type is the same
    println("\n=== Quantization Type ===")
    println("Q4_K type code: ", Int(GGUF.GGML_TYPE_Q4_K))
end

investigate_dequantization()
