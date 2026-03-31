using Inferno
using Inferno.GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check raw tensor dimensions for attention layer 3 (0-indexed)
prefix = "blk.3"
println("=== Attention tensors for layer 3 ===")
for name in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]
    tensor_name = "$prefix.$name"
    if haskey(file.tensors, tensor_name)
        info = file.tensors[tensor_name]
        println("$name: dimensions = ", info.dimensions, ", type = ", info.type)
    end
end

# Also check if there's q_norm and k_norm
println("\n=== QK Norm tensors ===")
for name in ["attn_q_a.norm", "attn_q_b.norm", "attn_k_a.norm", "attn_k_b.norm"]
    tensor_name = "$prefix.$name"
    if haskey(file.tensors, tensor_name)
        info = file.tensors[tensor_name]
        println("$name: dimensions = ", info.dimensions, ", type = ", info.type)
    else
        println("$name: NOT FOUND")
    end
end
