using Inferno
using LinearAlgebra

function trace_prompt_processing()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches and reset states
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process prompt "The capital of France is"
    prompt_tokens = [761, 6512, 315, 9339, 370]
    
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(prompt_tokens)
        println("\n=== Token $pos: \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [tok])), "\" (id=$tok) ===")
        
        # Get embedding
        x = model.embed[:, tok]
        println("Embedding norm: ", sqrt(sum(abs2.(x))))
        
        # Process through layers
        for (i, layer) in enumerate(model.layers)
            x_before = copy(x)
            x = layer(x, pos-1, model.rope, caches[i])
            
            if i == 1 || i == 24
                println("  After layer $i: norm=", sqrt(sum(abs2.(x))), ", delta=", sqrt(sum(abs2.(x - x_before))))
            end
        end
        
        println("Final hidden norm: ", sqrt(sum(abs2.(x))))
    end
    
    # Get logits
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    
    println("\n=== Final logits ===")
    println("Max: ", maximum(logits))
    println("Min: ", minimum(logits))
    
    # Top predictions
    top_indices = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 predictions:")
    for idx in top_indices
        println("  $(idx-1) -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx-1])), "\": ", round(logits[idx], digits=3))
    end
end

trace_prompt_processing()
