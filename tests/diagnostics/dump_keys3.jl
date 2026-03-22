include("src/GGUF.jl")
using .GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
println("Has blk.0.ssm_a? ", haskey(file.tensors, "blk.0.ssm_a"))
for k in keys(file.tensors)
    if occursin("ssm_a", k)
        println("FOUND SSM_A: ", k)
    end
end
