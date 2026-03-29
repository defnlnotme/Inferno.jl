using Inferno

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check SSM parameters for a layer
for (i, layer) in enumerate(model.layers)
    if layer.is_ssm
        println("Layer $i (SSM):")
        println("  ssm_a: ", layer.op.ssm_a)
        println("  ssm_dt_bias: ", layer.op.ssm_dt_bias)
        println()
        if i >= 3
            break
        end
    end
end
