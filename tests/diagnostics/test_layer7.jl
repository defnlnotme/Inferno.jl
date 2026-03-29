using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

    h = copy(model.embed[:, 761])
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    for i in 1:7
        layer = model.layers[i]
        h_before = copy(h)
        h = layer(h, 0, model.rope, caches[i])
        println("Layer $i ($(layer.is_ssm ? "SSM" : "Attn")):")
        println("  Input norm: $(round(sqrt(sum(abs2, h_before)), digits=3))")
        println("  Output norm: $(round(sqrt(sum(abs2, h)), digits=3))")
        println("  Output max: $(round(maximum(abs.(h)), digits=3))")
        println("  Output min: $(round(minimum(h), digits=3))")
    end
end

main()
