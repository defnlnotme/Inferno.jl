include("src/GGUF.jl")
using .GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
println("Has ssm_norm? ", haskey(file.tensors, "blk.0.ssm_norm.weight"))
if haskey(file.tensors, "blk.0.ssm_norm.weight")
    info = file.tensors["blk.0.ssm_norm.weight"]
    println("ssm_norm dims: ", info.dimensions)
end
