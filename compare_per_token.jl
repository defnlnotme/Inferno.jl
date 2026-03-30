using Inferno
using LinearAlgebra

function compare_per_token()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    println("=== Layer 1 output norms per token ===")
    
    prompt_tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    for (pos, tok) in enumerate(prompt_tokens)
        # Process each token independently (fresh state)
        caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
        Inferno.ModelCPU.reset_states_cpu!(model)
        
        x = model.embed[:, tok]
        x = model.layers[1](x, pos-1, model.rope, caches[1])
        
        println("Token $pos ($tok): norm = ", round(sqrt(sum(abs2.(x))), digits=3))
    end
    
    println("\n=== Accumulated state (all previous tokens) ===")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    for (pos, tok) in enumerate(prompt_tokens)
        x = model.embed[:, tok]
        x = model.layers[1](x, pos-1, model.rope, caches[1])
        println("Token $pos: norm = ", round(sqrt(sum(abs2.(x))), digits=3))
    end
end

compare_per_token()
