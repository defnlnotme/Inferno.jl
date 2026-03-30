using Inferno
using LinearAlgebra

function map_positions()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    start = Int(file.data_offset + gate_info.offset) + 1
    num_elements = Int(prod(gate_info.dimensions))
    
    weights_raw = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    mlp6 = model.layers[6].mlp
    
    println("=== Mapping Positions ===")
    
    # For each position in gate_weight, find the corresponding position in weights_raw
    # gate_weight has shape (3584, 1024)
    # weights_raw has length 1024 * 3584 = 3670016
    
    # In GGUF (row-major), the tensor [1024, 3584] is stored as:
    # Element [i, j] is at position i * 3584 + j (0-indexed)
    # Or (i+1) * 3584 + (j+1) (1-indexed, Julia style)
    
    # Wait, let me think about this differently
    # The GGUF tensor is [1024, 3584] = 1024 rows, 3584 columns (C-style)
    # In row-major storage, data is stored row by row
    # So position 1 is [0, 0], position 2 is [0, 1], ..., position 3584 is [0, 3583]
    # Position 3585 is [1, 0], etc.
    
    # In GGUF convention, the tensor represents a matrix where:
    # - For FFN gate: input size = 1024 (hidden), output size = 3584 (intermediate)
    # - Matrix multiply: output = weight * input
    # - So weight should be (3584, 1024) in Julia
    
    # The GGUF shape [1024, 3584] in C-style means:
    # - weight_C[i][j] accesses element at row i, column j
    # - Row i contains the weights for output neuron i
    # - Column j contains the weights for input neuron j
    # - So weight_C is the TRANSPOSE of what we want in Julia
    
    # In Julia, we want weight_J such that output = weight_J * input
    # weight_J should be (3584, 1024)
    # weight_J[i, j] = weight_C[j-1][i-1] (converting from 0-indexed to 1-indexed)
    #                = weights_raw[(j-1) * 3584 + (i-1) + 1]
    #                = weights_raw[(j-1) * 3584 + i]
    
    # So for gate_weight[i, j] (1-indexed, Julia):
    # gate_weight[i, j] should be weights_raw[(j-1) * 3584 + i]
    
    # Let me verify:
    i, j = 1, 1
    expected_pos = (j - 1) * 3584 + i
    println("gate_weight[$i, $j] = ", round(mlp6.gate_weight[i, j], digits=5))
    println("Expected position in weights_raw: $expected_pos")
    println("weights_raw[$expected_pos] = ", round(weights_raw[expected_pos], digits=5))
    
    if abs(mlp6.gate_weight[i, j] - weights_raw[expected_pos]) < 0.001
        println("MATCH!")
    else
        println("MISMATCH!")
    end
    
    # Try other positions
    println("\n=== Testing other positions ===")
    for (i, j) in [(1, 1), (1, 2), (2, 1), (100, 100), (1000, 500)]
        expected_pos = (j - 1) * 3584 + i
        if expected_pos <= length(weights_raw)
            match = abs(mlp6.gate_weight[i, j] - weights_raw[expected_pos]) < 0.001
            println("gate_weight[$i, $j] = $(round(mlp6.gate_weight[i, j], digits=5)), weights_raw[$expected_pos] = $(round(weights_raw[expected_pos], digits=5)) -> $(match ? "MATCH" : "MISMATCH")")
        end
    end
end

map_positions()
