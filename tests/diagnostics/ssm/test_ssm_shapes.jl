#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Check all SSM tensors for layer 0
println("Layer 0 SSM tensors:")
for (name, tensor) in file.tensors
    if startswith(String(name), "blk.0.ssm") || startswith(String(name), "blk.0.attn")
        println("  $name: $(tensor.dimensions)")
    end
end

println("\n\nLayer 0 tensor shapes after extraction:")
# Check shapes after extraction
for name in ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
             "blk.0.ssm_conv1d.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
             "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight"]
    if haskey(file.tensors, name)
        raw = Inferno.LoaderCPU.extract_tensor_cpu(file, name)
        println("  $name raw: $(size(raw))")
        println("  $name transposed: $(size(raw'))")
    else
        println("  $name: NOT FOUND")
    end
end
