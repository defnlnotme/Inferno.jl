using Inferno
using LinearAlgebra

function trace_second_token()
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token
    println("=== Processing first token (\"The\") ===")
    x = model.embed[:, 761]
    println("Initial embedding norm: ", sqrt(sum(abs2.(x))))
    
    for (i, layer) in enumerate(model.layers)
        x = layer(x, 0, model.rope, caches[i])
    end
    
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    
    top_idx = sortperm(logits, rev=true)[1]
    println("Top prediction: $top_idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [top_idx])), "\"")
    
    # Process second token
    println("\n=== Processing second token (\" capital\") ===")
    x2 = model.embed[:, 6512]
    println("Initial embedding norm: ", sqrt(sum(abs2.(x2))))
    
    for (i, layer) in enumerate(model.layers)
        x2 = layer(x2, 1, model.rope, caches[i])
        
        if any(isnan.(x2)) || any(isinf.(x2))
            println("ERROR: NaN/Inf after layer $i")
            break
        end
    end
    
    x2_normed = model.final_norm(x2)
    logits2 = model.lm_head * x2_normed
    
    println("Logits max: ", maximum(logits2))
    println("Logits min: ", minimum(logits2))
    
    top_indices = sortperm(logits2, rev=true)[1:10]
    println("\nTop 10 predictions:")
    for idx in top_indices
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits2[idx], digits=3), ")")
    end
end

trace_second_token()
