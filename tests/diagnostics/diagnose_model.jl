using Inferno
using Inferno.GGUF

function diagnose(model_path)
    println("Diagnosing model: ", model_path)
    file = GGUF.read_gguf(model_path)
    
    println("\nMetadata:")
    for (k, v) in file.metadata
        if k in ["general.architecture", "qwen2.attention.head_count", "qwen2.attention.key_length", "qwen2_5.attention.head_count"]
            println("  $k: $v")
        end
    end
    
    arch = get(file.metadata, "general.architecture", "qwen2")
    println("\nDetected Architecture: ", arch)
    
    println("\nTensor Type Distribution:")
    types = Dict{Inferno.GGUF.GGMLType, Int}()

    for (name, info) in file.tensors
        types[info.type] = get(types, info.type, 0) + 1
        if name == "output.weight" || name == "token_embd.weight"
             println("  $name: $(info.dimensions) (type: $(info.type))")
        end
    end
    for (t, c) in types
        println("  $t: $c")
    end

    println("\nSample Tensors (first 10):")
    count = 0
    for (name, info) in file.tensors
        println("  $name: $(info.dimensions) (type: $(info.type))")
        count += 1
        if count >= 10 break end
    end

end

model_path = length(ARGS) > 0 ? ARGS[1] : get(ENV, "INFERNO_MODEL", "unsloth/Qwen3.5-0.8B-GGUF")
diagnose(model_path)
flush(stdout)
