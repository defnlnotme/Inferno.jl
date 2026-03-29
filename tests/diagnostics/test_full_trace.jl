using Inferno
using Statistics

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Test with proper prompt
    prompt = "The capital of France"
    
    # Tokenize manually - we know the tokens
    # "The" = 760, " capital" = 6511, " of" = 314, " France" = 9338
    tokens = [760, 6511, 314, 9338]

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    h = nothing
    for (pos, tok) in enumerate(tokens)
        println("\n=== Position $pos, token $tok ===")
        
        h = copy(model.embed[:, tok])
        println("Embed: mean=$(mean(h)), std=$(std(h))")

        for (i, layer) in enumerate(model.layers)
            h_before = copy(h)
            h = layer(h, pos-1, model.rope, caches[i])
            
            # Check for issues
            if !isfinite(sum(h))
                println("ERROR: Layer $i produced non-finite values!")
                break
            end
            
            # Track signal growth
            if i % 4 == 0 || i == length(model.layers)
                println("  Layer $i ($(layer.is_ssm ? "SSM" : "Attn")): mean=$(mean(h)), std=$(std(h)), max=$(maximum(abs.(h)))")
            end
        end
        
        if !isfinite(sum(h))
            break
        end
    end

    if h !== nothing && isfinite(sum(h))
        h_normed = model.final_norm(h)
        println("\nFinal norm: mean=$(mean(h_normed)), std=$(std(h_normed))")
        
        logits = model.lm_head * h_normed
        println("Logits: mean=$(mean(logits)), std=$(std(logits)), max=$(maximum(logits)), min=$(minimum(logits))")
        
        top10 = sortperm(logits, rev=true)[1:10]
        println("\nTop 10 tokens after prompt '$prompt':")
        file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
        tokens_list = file.metadata["tokenizer.ggml.tokens"]
        for t in top10
            println("  Token $t: $(repr(tokens_list[t+1])) (logit=$(logits[t]))")
        end
        
        # Check Paris specifically
        paris_token = 11751
        println("\n'Paris' token ($paris_token) logit: $(logits[paris_token])")
    end
end

main()
