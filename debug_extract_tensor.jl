using Inferno
using LinearAlgebra

function debug_extract_tensor()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Manually call extract_tensor_cpu
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    
    # Trace through extract_tensor_cpu
    num_elements = Int(prod(gate_info.dimensions))
    start = Int(file.data_offset + gate_info.offset) + 1
    
    dims = Tuple(Int.(gate_info.dimensions))
    inner = dims[1]  # 1024
    outer = dims[2]  # 3584
    
    println("=== extract_tensor_cpu debug ===")
    println("dims: $dims")
    println("inner: $inner")
    println("outer: $outer")
    println("num_elements: $num_elements")
    println("start: $start")
    
    # Dequantize
    data = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    println("\nDequantized data length: ", length(data))
    println("First 10 values: ", round.(data[1:10], digits=5))
    
    # Reshape (our fix)
    M = reshape(data, outer, inner)'
    println("\nAfter reshape(data, $outer, $inner)':")
    println("  Shape: ", size(M))
    println("  M[1, 1:5]: ", round.(M[1, 1:5], digits=5))
    
    # This is what extract_tensor_cpu returns
    
    # Now in MLP loading, we transpose again
    gate_weight = Matrix(Float32.(M'))
    println("\nAfter transpose (in MLP loading):")
    println("  Shape: ", size(gate_weight))
    println("  gate_weight[1, 1:5]: ", round.(gate_weight[1, 1:5], digits=5))
    
    # Load actual model
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    mlp6 = model.layers[6].mlp
    
    println("\nActual loaded gate_weight:")
    println("  Shape: ", size(mlp6.gate_weight))
    println("  gate_weight[1, 1:5]: ", round.(mlp6.gate_weight[1, 1:5], digits=5))
    
    # Compare
    println("\n=== Comparison ===")
    if gate_weight ≈ mlp6.gate_weight
        println("Manual and loaded gate_weight MATCH!")
    else
        println("Manual and loaded gate_weight MISMATCH!")
    end
end

debug_extract_tensor()
