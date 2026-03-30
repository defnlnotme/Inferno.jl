using Inferno
using LinearAlgebra

function detailed_trace()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches and reset states
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token
    x = model.embed[:, 761]  # "The"
    pos = 0
    
    println("=== Processing token 'The' at position 0 ===")
    println("Input norm: ", sqrt(sum(abs2.(x))))
    
    # Process through layers with detailed logging
    for (i, layer) in enumerate(model.layers)
        x_before = copy(x)
        
        if layer.is_ssm
            println("\n--- Layer $i (SSM) ---")
            ssm = layer.op
            
            # Check initial state
            println("SSM state norm before: ", sqrt(sum(abs2.(ssm.h))))
            
            # Forward pass
            x = layer(x, pos, model.rope, caches[i])
            
            println("SSM state norm after: ", sqrt(sum(abs2.(ssm.h))))
            println("Output norm: ", sqrt(sum(abs2.(x))))
            println("Delta norm: ", sqrt(sum(abs2.(x - x_before))))
        else
            println("\n--- Layer $i (Attention) ---")
            x = layer(x, pos, model.rope, caches[i])
            println("Output norm: ", sqrt(sum(abs2.(x))))
            println("Delta norm: ", sqrt(sum(abs2.(x - x_before))))
        end
        
        if any(isnan.(x)) || any(isinf.(x))
            println("ERROR: NaN/Inf after layer $i")
            break
        end
    end
    
    # Final norm and logits
    println("\n=== Final output ===")
    println("Hidden norm before final norm: ", sqrt(sum(abs2.(x))))
    x_normed = model.final_norm(x)
    println("Hidden norm after final norm: ", sqrt(sum(abs2.(x_normed))))
    
    logits = model.lm_head * x_normed
    println("Logits max: ", maximum(logits))
    println("Logits min: ", minimum(logits))
    
    # Top predictions
    top_indices = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 predictions:")
    for idx in top_indices
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits[idx], digits=3), ")")
    end
end

detailed_trace()
