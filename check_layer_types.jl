using Inferno

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("=== Layer Types ===")
for (i, layer) in enumerate(model.layers)
    println("Layer $i: ", layer.is_ssm ? "SSM" : "Attention")
end
