using Inferno
using LinearAlgebra

function trace_second_token()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches and reset states
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token
    x = model.embed[:, 761]  # "The"
    pos = 0
    
    for (i, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, caches[i])
    end
    
    # Get first prediction
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    top_idx = sortperm(logits, rev=true)[1]
    println("First prediction: $top_idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [top_idx])), "\"")
    
    # Process second token (space)
    println("\n=== Processing second token (space) ===")
    x2 = model.embed[:, 221]  # space
    pos2 = 1
    
    for (i, layer) in enumerate(model.layers)
        x2_before = copy(x2)
        
        if layer.is_ssm
            ssm = layer.op
            println("\n--- Layer $i (SSM) ---")
            println("SSM state norm before: ", sqrt(sum(abs2.(ssm.h))))
            
            x2 = layer(x2, pos2, model.rope, caches[i])
            
            println("SSM state norm after: ", sqrt(sum(abs2.(ssm.h))))
            println("Output norm: ", sqrt(sum(abs2.(x2))))
            println("Delta from input: ", sqrt(sum(abs2.(x2 - x2_before))))
        else
            println("\n--- Layer $i (Attention) ---")
            x2 = layer(x2, pos2, model.rope, caches[i])
            println("Output norm: ", sqrt(sum(abs2.(x2))))
        end
    end
    
    # Get second prediction
    x2_normed = model.final_norm(x2)
    logits2 = model.lm_head * x2_normed
    
    println("\n=== Second prediction ===")
    println("Logits max: ", maximum(logits2))
    println("Logits min: ", minimum(logits2))
    
    top_indices = sortperm(logits2, rev=true)[1:10]
    println("\nTop 10 predictions:")
    for idx in top_indices
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits2[idx], digits=3), ")")
    end
end

trace_second_token()
