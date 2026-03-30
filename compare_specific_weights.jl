using Inferno
using LinearAlgebra

function compare_specific_weights()
    # Compare actual dequantized weight values
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    mlp6 = model.layers[6].mlp
    
    println("=== Specific Weight Value Comparison ===")
    
    # Sample some specific weights
    println("\ngate_weight first 10 values of row 1:")
    println(round.(mlp6.gate_weight[1, 1:10], digits=5))
    
    println("\nup_weight first 10 values of row 1:")
    println(round.(mlp6.up_weight[1, 1:10], digits=5))
    
    # Check if there's a systematic scaling difference
    # Let me compute the ratio of weight values
    gate_sample = mlp6.gate_weight[1:10, 1:10]
    up_sample = mlp6.up_weight[1:10, 1:10]
    
    println("\n=== Weight Sample Analysis ===")
    println("gate_weight [1:10, 1:10] stats:")
    println("  min: ", minimum(gate_sample))
    println("  max: ", maximum(gate_sample))
    println("  mean: ", round(sum(gate_sample) / length(gate_sample), digits=5))
    
    println("\nup_weight [1:10, 1:10] stats:")
    println("  min: ", minimum(up_sample))
    println("  max: ", maximum(up_sample))
    println("  mean: ", round(sum(up_sample) / length(up_sample), digits=5))
    
    # Check the ratio of element-wise values
    ratio_sample = gate_sample ./ up_sample
    println("\ngate/up ratio [1:10, 1:10]:")
    println("  mean: ", round(sum(ratio_sample) / length(ratio_sample), digits=3))
    println("  min: ", round(minimum(ratio_sample), digits=3))
    println("  max: ", round(maximum(ratio_sample), digits=3))
    
    # Now let me check if the issue is in the weight loading
    # by examining the raw GGUF data
    
    println("\n=== Raw GGUF Data Check ===")
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    up_info = file.tensors["blk.6.ffn_up.weight"]
    
    println("gate tensor offset: ", gate_info.offset)
    println("gate tensor elements: ", gate_info.ne)
    println("up tensor offset: ", up_info.offset)
    println("up tensor elements: ", up_info.ne)
    
    # The issue might be in how we extract the tensor
    # Let me check the extraction function
end

compare_specific_weights()
