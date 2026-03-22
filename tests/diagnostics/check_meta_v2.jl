include("src/GGUF.jl")
using .GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
arch = get(file.metadata, "general.architecture", "UNKNOWN")
println("Architecture: ", arch)
for k in keys(file.metadata)
    if occursin("rope", k)
        println("RoPE metadata: ", k, " = ", file.metadata[k])
    end
end
println("Full metadata keys related to architecture:")
for k in keys(file.metadata)
    if occursin(arch, k) && occursin("rope", k)
        println(k, " = ", file.metadata[k])
    end
end
