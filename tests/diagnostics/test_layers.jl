using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    println("Layer types:")
    for (i, layer) in enumerate(model.layers)
        println("  Layer $i: ", typeof(layer.op))
    end
    
    println("\nModel architecture: ", model.config.architecture)
    println("SSM conv kernel: ", model.config.ssm_conv_kernel)
end

main()
