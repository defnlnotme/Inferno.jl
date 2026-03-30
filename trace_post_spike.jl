using Inferno
using LinearAlgebra

function trace_post_spike()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    
    # Process through all layers, tracking state after each
    for i in 1:model.config.num_hidden_layers
        layer = model.layers[i]
        layer_type = layer.is_ssm ? "SSM" : "Attn"
        
        if layer.is_ssm
            ssm = layer.op
            state_before = sqrt(sum(abs2.(ssm.h)))
        end
        
        x_before = copy(x)
        x = layer(x, pos, model.rope, caches[i])
        
        if layer.is_ssm
            ssm = layer.op
            state_after = sqrt(sum(abs2.(ssm.h)))
            println("Layer $i ($layer_type): output norm = $(round(sqrt(sum(abs2.(x))), digits=3)), state norm before = $(round(state_before, digits=3)), after = $(round(state_after, digits=3))")
        else
            println("Layer $i ($layer_type): output norm = $(round(sqrt(sum(abs2.(x))), digits=3))")
        end
    end
    
    println("\nFinal output norm: ", round(sqrt(sum(abs2.(x))), digits=3))
end

trace_post_spike()
