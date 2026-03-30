using Inferno
using LinearAlgebra

function trace_weight_loading()
    # Load the model
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    mlp6 = model.layers[6].mlp
    
    # Check gate_weight values
    println("=== gate_weight values ===")
    println("Shape: ", size(mlp6.gate_weight))
    
    # The first row of gate_weight should correspond to:
    # - Row 0 of the GGUF tensor (after transpose)
    # - Which is the first 3584 values in the dequantized array
    
    println("\nFirst row (first 10 values):")
    println(round.(mlp6.gate_weight[1, 1:10], digits=5))
    
    # Load the raw data and dequantize
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    start = Int(file.data_offset + gate_info.offset) + 1
    num_elements = Int(prod(gate_info.dimensions))
    
    weights_raw = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    
    println("\nRaw dequantized weights (first 10 values):")
    println(round.(weights_raw[1:10], digits=5))
    
    # The first row of gate_weight should be the first 1024 values of the raw weights
    # (after transpose: we have (1024, 3584), then transpose to (3584, 1024))
    # Wait, let me trace through the reshaping...
    
    # GGUF dims: [1024, 3584]
    # After dequantize: flat array of 1024 * 3584 values in row-major order
    # 
    # In extract_tensor_cpu:
    #   inner = 1024, outer = 3584
    #   reshape(data, outer, inner)' = reshape(data, 3584, 1024)'
    #   = matrix of shape (1024, 3584)
    #
    # The reshape(data, 3584, 1024) creates:
    #   - Column 1 = values[1:3584] = first 3584 values
    #   - Column 2 = values[3585:7168] = next 3584 values
    #   - etc.
    #
    # After transpose:
    #   - Row 1 = values[1:3584] = first 3584 values
    #   - Row 2 = values[3585:7168] = next 3584 values
    #   - etc.
    #
    # In MLP loading:
    #   gate_weight = extract_tensor_cpu(...)'
    #   = (3584, 1024)
    #   - Column 1 = values[1:3584] = first 3584 values
    #   - etc.
    #
    # So gate_weight[1, :] = values[1:1024] (first 1024 values)
    # That's row 1 = column 1 of the (1024, 3584) matrix = values[1:3584]
    # Hmm, this is getting confusing...
    
    println("\n=== Detailed Trace ===")
    println("GGUF dims: [1024, 3584]")
    println("After dequantize: flat array of $(length(weights_raw)) values")
    println("")
    println("In extract_tensor_cpu:")
    println("  reshape(data, 3584, 1024) creates a (3584, 1024) matrix")
    println("  Column 1 = values[1:3584]")
    println("  Column 2 = values[3585:7168]")
    println("  ...")
    println("  Column 1024 = values[1023*3584+1:1024*3584]")
    println("")
    println("After transpose:")
    println("  Row 1 = values[1:3584] (first 3584 values)")
    println("  Row 2 = values[3585:7168] (next 3584 values)")
    println("")
    println("In MLP loading (transpose again):")
    println("  gate_weight = extract_tensor_cpu(...)'")
    println("  Shape: (3584, 1024)")
    println("  Column 1 = values[1:3584]")
    println("  Row 1 = values[1:1024] (first 1024 values)")
    println("")
    println("So gate_weight[1, :] should be values[1:1024]")
    println("Actual gate_weight[1, 1:10]: ", round.(mlp6.gate_weight[1, 1:10], digits=5))
    println("Expected values[1:10]: ", round.(weights_raw[1:10], digits=5))
    
    if all(abs.(mlp6.gate_weight[1, 1:10] .- weights_raw[1:10]) .< 0.001)
        println("MATCH!")
    else
        println("MISMATCH!")
    end
end

trace_weight_loading()
