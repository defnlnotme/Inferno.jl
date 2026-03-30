using Inferno
using LinearAlgebra

function manual_dequantize_first_block()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    gate_info = file.tensors["blk.6.ffn_gate.weight"]
    start = Int(file.data_offset + gate_info.offset) + 1
    
    # Manual dequantization of first block
    # Q4_K block: d (2), dmin (2), scales (12), qs (128)
    
    d = Float32(reinterpret(Float16, file.tensor_data[start:start+1])[1])
    dmin = Float32(reinterpret(Float16, file.tensor_data[start+2:start+3])[1])
    scales = file.tensor_data[start+4:start+15]
    qs = file.tensor_data[start+16:start+143]
    
    println("=== Manual Dequantization of First Block ===")
    println("d = ", d)
    println("dmin = ", dmin)
    println("scales = ", scales)
    
    # Dequantize using our implementation
    weights = Inferno.Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), 256)
    
    println("\nFirst 20 weights from our dequantization:")
    println(round.(weights[1:20], digits=5))
    
    # Manual dequantization for comparison
    # For each super-group (4 groups of 64 elements)
    println("\n=== Manual calculation ===")
    
    for j in 0:3
        is_idx = 2 * j
        
        # Get scales for this super-group
        if is_idx < 4
            sc1 = scales[is_idx + 1] & 63
            m1 = scales[is_idx + 5] & 63
        else
            sc1 = (scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4)
            m1 = (scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4)
        end
        
        if (is_idx + 1) < 4
            sc2 = scales[is_idx + 2] & 63
            m2 = scales[is_idx + 6] & 63
        else
            sc2 = (scales[is_idx + 6] & 0x0f) | ((scales[is_idx - 2] >> 6) << 4)
            m2 = (scales[is_idx + 6] >> 4) | ((scales[is_idx + 2] >> 6) << 4)
        end
        
        d1 = d * Float32(sc1)
        min1 = dmin * Float32(m1)
        d2 = d * Float32(sc2)
        min2 = dmin * Float32(m2)
        
        println("Super-group $j:")
        println("  sc1=$sc1, m1=$m1, d1=$d1, min1=$min1")
        println("  sc2=$sc2, m2=$m2, d2=$d2, min2=$min2")
        
        # Dequantize first few elements
        for l in 0:3
            ql_val = qs[j * 32 + l + 1]
            w1 = d1 * Float32(ql_val & 0x0f) - min1
            w2 = d2 * Float32(ql_val >> 4) - min2
            println("  element $l: q=$(ql_val), w1=$(round(w1, digits=5)), w2=$(round(w2, digits=5))")
        end
    end
end

manual_dequantize_first_block()
