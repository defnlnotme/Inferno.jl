using Inferno
using LinearAlgebra

function check_raw_data()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    up_info = file.tensors["blk.6.ffn_up.weight"]
    
    start_gate = Int(file.data_offset + gate_info.offset) + 1
    start_up = Int(file.data_offset + up_info.offset) + 1
    
    println("=== Raw Data Check ===")
    println("\ngate tensor:")
    println("  offset: ", gate_info.offset)
    println("  dimensions: ", gate_info.dimensions)
    println("  type: ", gate_info.type)
    println("  first 20 bytes: ", file.tensor_data[start_gate:start_gate+19])
    
    println("\nup tensor:")
    println("  offset: ", up_info.offset)
    println("  dimensions: ", up_info.dimensions)
    println("  type: ", up_info.type)
    println("  first 20 bytes: ", file.tensor_data[start_up:start_up+19])
    
    # Check if the raw data is different
    println("\n=== Raw Data Comparison ===")
    if gate_info.offset == up_info.offset
        println("gate and up have SAME offset - they're the same tensor!")
    else
        println("gate and up have different offsets")
    end
    
    # Check first few blocks
    # Q4_K block is 144 bytes
    # Each block has 256 elements
    
    println("\n=== First Block Analysis ===")
    println("Q4_K block structure:")
    println("  d (scale): 2 bytes (Float16)")
    println("  dmin (min scale): 2 bytes (Float16)")
    println("  scales: 12 bytes")
    println("  qs (quantized values): 128 bytes")
    println("  Total: 144 bytes")
    
    # Read first block of gate
    d_bytes = file.tensor_data[start_gate:start_gate+1]
    d = reinterpret(Float16, d_bytes)[1]
    println("\ngate first block:")
    println("  d (scale): ", Float32(d))
    
    dmin_bytes = file.tensor_data[start_gate+2:start_gate+3]
    dmin = reinterpret(Float16, dmin_bytes)[1]
    println("  dmin: ", Float32(dmin))
    
    scales = file.tensor_data[start_gate+4:start_gate+15]
    println("  scales: ", scales)
end

check_raw_data()
