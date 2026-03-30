using Inferno
using LinearAlgebra

function trace_reshape()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    start = Int(file.data_offset + gate_info.offset) + 1
    num_elements = Int(prod(gate_info.dimensions))
    
    data = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    
    println("=== Raw data ===")
    println("Length: ", length(data))
    println("data[1] = ", round(data[1], digits=5))
    println("data[2] = ", round(data[2], digits=5))
    println("data[3] = ", round(data[3], digits=5))
    
    # My fix
    inner = 1024
    outer = 3584
    M = reshape(data, outer, inner)'
    
    println("\n=== After reshape(data, outer, inner)' ===")
    println("Shape: ", size(M))
    println("M[1, 1] = ", round(M[1, 1], digits=5), " (expected data[1] = ", round(data[1], digits=5), ")")
    println("M[1, 2] = ", round(M[1, 2], digits=5), " (expected data[2] = ", round(data[2], digits=5), ")")
    println("M[2, 1] = ", round(M[2, 1], digits=5), " (expected data[3585])")
    
    # Check if M[1, 1] == data[1]
    if M[1, 1] ≈ data[1]
        println("\nM[1, 1] matches data[1]!")
    else
        println("\nM[1, 1] does NOT match data[1]!")
        println("M[1, 1] = ", M[1, 1])
        println("data[1] = ", data[1])
    end
    
    # Let me check what reshape does
    # In Julia, reshape(data, 3584, 1024) creates a column-major matrix
    # where data[1:3584] becomes column 1, etc.
    M_raw = reshape(data, outer, inner)
    println("\n=== reshape(data, outer, inner) (no transpose) ===")
    println("Shape: ", size(M_raw))
    println("M_raw[1, 1] = ", round(M_raw[1, 1], digits=5))
    println("M_raw[2, 1] = ", round(M_raw[2, 1], digits=5))
    println("M_raw[1, 2] = ", round(M_raw[1, 2], digits=5))
    
    # After transpose
    println("\n=== After transpose ===")
    println("M'[1, 1] = M[1, 1] = data[1] = ", round(data[1], digits=5))
    println("M'[1, 2] = M[2, 1] = data[2] = ", round(data[2], digits=5))
    println("M'[2, 1] = M[1, 2] = data[3585] = ", round(data[3585], digits=5))
end

trace_reshape()
