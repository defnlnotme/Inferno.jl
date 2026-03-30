using Inferno
using LinearAlgebra

function find_value_positions()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    start = Int(file.data_offset + gate_info.offset) + 1
    num_elements = Int(prod(gate_info.dimensions))
    
    weights_raw = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    mlp6 = model.layers[6].mlp
    
    println("=== Finding Position of Values ===")
    println("gate_weight[1, 1] = ", round(mlp6.gate_weight[1, 1], digits=5))
    println("gate_weight[1, 2] = ", round(mlp6.gate_weight[1, 2], digits=5))
    println("gate_weight[1, 3] = ", round(mlp6.gate_weight[1, 3], digits=5))
    
    # Search for these values in the raw weights
    println("\nSearching for gate_weight[1, 1] = ", round(mlp6.gate_weight[1, 1], digits=5))
    for i in 1:min(10000, length(weights_raw))
        if abs(weights_raw[i] - mlp6.gate_weight[1, 1]) < 0.001
            println("  Found at position $i")
        end
    end
    
    # Let's check specific positions
    println("\n=== Checking specific positions ===")
    println("weights_raw[1] = ", round(weights_raw[1], digits=5))
    println("weights_raw[1024] = ", round(weights_raw[1024], digits=5))
    println("weights_raw[1025] = ", round(weights_raw[1025], digits=5))
    println("weights_raw[3584] = ", round(weights_raw[3584], digits=5))
    println("weights_raw[3585] = ", round(weights_raw[3585], digits=5))
    
    # The issue might be in how we're transposing
    # Let me check the actual matrix we get
    M = reshape(weights_raw, 3584, 1024)'
    println("\n=== Matrix from reshape(3584, 1024)' ===")
    println("M[1, 1:5] = ", round.(M[1, 1:5], digits=5))
    println("M[2, 1:5] = ", round.(M[2, 1:5], digits=5))
    
    # And after another transpose
    M2 = M'
    println("\n=== Matrix M' ===")
    println("M2[1, 1:5] = ", round.(M2[1, 1:5], digits=5))
    println("M2[2, 1:5] = ", round.(M2[2, 1:5], digits=5))
    
    # Compare with gate_weight
    println("\n=== gate_weight ===")
    println("gate_weight[1, 1:5] = ", round.(mlp6.gate_weight[1, 1:5], digits=5))
    println("gate_weight[2, 1:5] = ", round.(mlp6.gate_weight[2, 1:5], digits=5))
end

find_value_positions()
