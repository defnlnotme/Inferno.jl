using Inferno
using LinearAlgebra

function check_layer7_accumulation()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    prompt_tokens = [761, 6512, 315, 9339, 370]
    
    println("=== Layer 7 output norms ===")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    for (pos, tok) in enumerate(prompt_tokens)
        x = model.embed[:, tok]
        for i in 1:7
            x = model.layers[i](x, pos-1, model.rope, caches[i])
        end
        println("Token $pos: norm = ", round(sqrt(sum(abs2.(x))), digits=3))
    end
    
    println("\n=== SSM state check for layer 7 ===")
    ssm = model.layers[7].op
    println("SSM h state norm after all tokens: ", round(sqrt(sum(abs2.(ssm.h))), digits=3))
end

check_layer7_accumulation()
