using Inferno
using LinearAlgebra

function trace_attention()
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Prompt tokens
    tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    # Process all prompt tokens
    for (pos, tok) in enumerate(tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    
    # Check KV cache for attention layers
    println("KV cache after prompt:")
    attn_layers = [i for (i, layer) in enumerate(model.layers) if !layer.is_ssm]
    
    for i in attn_layers
        layer = model.layers[i]
        cache = caches[i]
        println("\nLayer $i (Attention):")
        println("  K shape: ", size(cache.k))
        println("  V shape: ", size(cache.v))
        
        # Check norms at each position
        for p in 1:5
            k_norm = sqrt(sum(abs2.(cache.k[:, :, p])))
            v_norm = sqrt(sum(abs2.(cache.v[:, :, p])))
            println("  Position $p K norm: ", k_norm, ", V norm: ", v_norm)
        end
    end
end

trace_attention()
