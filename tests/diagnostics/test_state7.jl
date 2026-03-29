using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

    h = copy(model.embed[:, 761])
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    for i in 1:7
        layer = model.layers[i]
        h = layer(h, 0, model.rope, caches[i])
        
        if layer.is_ssm
            ssm = layer.op
            state = caches[i]
            println("Layer $i (SSM) state:")
            println("  State norm: $(round(sqrt(sum(abs2, state.state)), digits=3))")
            println("  State max: $(round(maximum(abs.(state.state)), digits=3))")
            println("  State size: $(size(state.state))")
        end
    end
end

main()
