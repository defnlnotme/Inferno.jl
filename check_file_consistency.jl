using Inferno
using LinearAlgebra

function check_file_consistency()
    # Check if the model loading is using the same file
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    println("gate_info.offset = ", gate_info.offset)
    println("gate_info.dimensions = ", gate_info.dimensions)
    println("gate_info.type = ", gate_info.type)
    
    # Load model
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Compare with what we expect
    mlp6 = model.layers[6].mlp
    
    println("\nComparing gate_weight:")
    println("  Shape: ", size(mlp6.gate_weight))
    println("  First values: ", round.(mlp6.gate_weight[1, 1:10], digits=5))
    
    # Manually dequantize
    start = Int(file.data_offset + gate_info.offset) + 1
    num_elements = Int(prod(gate_info.dimensions))
    weights_raw = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    
    println("\nRaw dequantized:")
    println("  Length: ", length(weights_raw))
    println("  First values: ", round.(weights_raw[1:10], digits=5))
    
    # Check if the values are somewhere else in the file
    # Maybe we're reading from the wrong tensor
    
    # Find which tensor contains gate_weight[1, 1] = 0.0089
    target = mlp6.gate_weight[1, 1]
    println("\nSearching for $target in all tensors...")
    
    count = 0
    for (name, info) in file.tensors
        if occursin("blk.6", name) && info.type == Inferno.GGUF.GGML_TYPE_Q4_K
            t_start = Int(file.data_offset + info.offset) + 1
            t_num = Int(prod(info.dimensions))
            t_weights = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[t_start:end]), t_num)
            
            for i in 1:min(100, length(t_weights))
                if abs(t_weights[i] - target) < 0.001
                    println("  Found $target in $name at position $i")
                    count += 1
                    if count > 5
                        break
                    end
                end
            end
        end
    end
end

check_file_consistency()
