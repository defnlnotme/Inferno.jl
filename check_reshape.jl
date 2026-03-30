using Inferno
using LinearAlgebra

function check_reshape()
    # The dequantize function returns a flat vector
    # Then we reshape it to (inner, outer)
    
    # GGUF dimensions: [1024, 3584]
    # inner = 1024, outer = 3584
    # num_elements = 1024 * 3584 = 3,670,016
    
    # The dequantize function returns values in some order
    # We need to check if the reshape is correct
    
    # Let me trace through a simple example:
    # If we have weights stored as [1024, 3584] in column-major order
    # And we reshape to (1024, 3584), we get the same layout
    # Then transpose to (3584, 1024)
    
    # For matrix-vector multiply:
    # output[i] = sum_j weight[i, j] * input[j]
    # This is: output = weight * input
    
    # But in Julia, reshape(data, inner, outer) creates a matrix
    # where data[1:inner] becomes column 1, data[inner+1:2*inner] becomes column 2, etc.
    # This is column-major order
    
    # GGUF also stores data in row-major order? Let me check.
    
    println("=== GGUF Storage Order Check ===")
    println("")
    println("GGUF tensor dimensions: [1024, 3584]")
    println("This means: 1024 rows, 3584 columns (in row-major C convention)")
    println("")
    println("In GGUF (C-style), the data is stored row-major:")
    println("  data[0..3583] = row 0")
    println("  data[3584..7167] = row 1")
    println("  etc.")
    println("")
    println("In Julia, reshape(data, 1024, 3584) creates column-major:")
    println("  data[1..1024] = column 1")
    println("  data[1025..2048] = column 2")
    println("  etc.")
    println("")
    println("THIS IS THE BUG!")
    println("GGUF stores data row-major, but Julia reshape is column-major!")
    println("")
    println("The fix: reshape to (outer, inner) then transpose")
    println("Or: reshape(data, outer, inner)' to get correct layout")
    
    # Let me verify this theory
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    mlp6 = model.layers[6].mlp
    
    println("\n=== Current Weight Shape ===")
    println("gate_weight shape: ", size(mlp6.gate_weight))
    println("Expected: (3584, 1024) for intermediate_size x hidden_size")
end

check_reshape()
