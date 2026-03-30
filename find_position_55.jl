using Inferno
using LinearAlgebra

function find_position_55()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    start = Int(file.data_offset + gate_info.offset) + 1
    num_elements = Int(prod(gate_info.dimensions))
    
    weights_raw = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    
    println("=== Position Analysis ===")
    println("gate_weight[1, 1] = 0.0089")
    println("This value is at raw position 55")
    println("")
    
    # Position 55 is in block 0 (positions 1-256)
    # In the raw data, position 55 means:
    # - It's the 55th element of the first 256-element block
    # - Or position 54 (0-indexed)
    
    # In our reshape logic:
    # reshape(data, 3584, 1024) creates:
    # - Column 1: positions 1-3584
    # - Column 2: positions 3585-7168
    # - etc.
    # After transpose:
    # - Row 1: positions 1-3584
    # - Row 2: positions 3585-7168
    # - etc.
    # After another transpose (in MLP loading):
    # - Column 1: positions 1-3584
    # - Column 2: positions 3585-7168
    # - etc.
    # - Row 1: positions 1, 3585, 7169, ... (1 + i*3584)
    # - Row 2: positions 2, 3586, 7170, ...
    # - etc.
    
    # So gate_weight[1, 1] = position 1 + (1-1)*3584 = position 1
    # But the value at position 1 is -0.02933, not 0.0089!
    
    # Unless... the transpose is not doing what I think
    # Let me check
    
    M = reshape(weights_raw, 3584, 1024)'
    # M is (1024, 3584)
    # M[1, 1] = weights_raw[1]
    # M[1, 2] = weights_raw[2]
    # etc.
    
    # After transpose: M' is (3584, 1024)
    # M'[1, 1] = M[1, 1] = weights_raw[1]
    # M'[2, 1] = M[1, 2] = weights_raw[2]
    # M'[1, 2] = M[2, 1] = weights_raw[3585]
    # etc.
    
    # So M'[i, j] = weights_raw[(j-1)*3584 + i]
    # M'[1, 1] = weights_raw[1]
    # But gate_weight[1, 1] = 0.0089, and weights_raw[1] = -0.02933
    
    # Wait, maybe the issue is that we're NOT reading from position 55
    # Let me check weights_raw[55]
    
    println("weights_raw[55] = ", round(weights_raw[55], digits=5))
    
    # And position 55 in the matrix after transpose:
    # M'[55, 1] = weights_raw[55]
    # But gate_weight is (3584, 1024)
    # gate_weight[1, 1] should be at position...
    # If gate_weight = M', then gate_weight[1, 1] = M'[1, 1] = weights_raw[1]
    
    # But if there's a transpose in MLP loading, then:
    # gate_weight = M'' = reshape(data, 3584, 1024)
    # gate_weight[1, 1] = M'[1, 1] = weights_raw[1] = -0.02933
    
    # STILL doesn't match!
    
    # Let me check if the model loader is doing something else
    println("\nLet me check what position 55 corresponds to...")
    
    # Position 55 in a 3584x1024 matrix:
    # Column-major: position 55 = row 55, column 1 (since 55 < 3584)
    # So in reshape(data, 3584, 1024), position 55 is at row 55, column 1
    
    # In the transposed matrix M' (1024, 3584):
    # Position 55 would be at... row = 55 % 3584 = 55, col = 55 ÷ 3584 + 1 = 1
    # Wait, that's not right.
    
    # In column-major (1024, 3584):
    # Position 55 = col 1 (since 55 < 1024)
    # No wait, 55 > 1024? No, 55 < 1024.
    # So position 55 is at row 55, col 1 of the (1024, 3584) matrix
    
    # M[55, 1] = weights_raw[55]
    
    # But in M' (3584, 1024):
    # M'[55, 1] = M[1, 55] = weights_raw[(55-1)*3584 + 1]
    
    # Let me check that
    pos_in_M = (55 - 1) * 3584 + 1
    println("Position in M for M'[55, 1] = weights_raw[$pos_in_M]")
    if pos_in_M <= length(weights_raw)
        println("  weights_raw[$pos_in_M] = ", round(weights_raw[pos_in_M], digits=5))
    end
end

find_position_55()
