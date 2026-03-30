using Inferno
using LinearAlgebra

function trace_full_flow()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Call extract_tensor_cpu directly
    gate_tensor = Inferno.LoaderCPU.extract_tensor_cpu(file, "blk.6.ffn_gate.weight")
    
    println("=== Direct call to extract_tensor_cpu ===")
    println("Type: ", typeof(gate_tensor))
    println("Shape: ", size(gate_tensor))
    println("gate_tensor[1, 1:5]: ", round.(gate_tensor[1, 1:5], digits=5))
    
    # Now load model and check
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    mlp6 = model.layers[6].mlp
    
    println("\n=== Loaded model gate_weight ===")
    println("Type: ", typeof(mlp6.gate_weight))
    println("Shape: ", size(mlp6.gate_weight))
    println("gate_weight[1, 1:5]: ", round.(mlp6.gate_weight[1, 1:5], digits=5))
    
    # Check if they match
    println("\n=== Comparison ===")
    if gate_tensor ≈ mlp6.gate_weight
        println("MATCH!")
    else
        println("MISMATCH!")
        
        # Find where they differ
        diff_count = 0
        for i in 1:min(10, size(gate_tensor, 1))
            for j in 1:min(10, size(gate_tensor, 2))
                if abs(gate_tensor[i, j] - mlp6.gate_weight[i, j]) > 0.001
                    diff_count += 1
                    if diff_count <= 5
                        println("  gate_tensor[$i, $j] = $(round(gate_tensor[i, j], digits=5)), gate_weight[$i, $j] = $(round(mlp6.gate_weight[i, j], digits=5))")
                    end
                end
            end
        end
        println("  ... $diff_count differences found")
    end
end

trace_full_flow()
