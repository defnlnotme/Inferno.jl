using Inferno
using LinearAlgebra

function compare_with_llamacpp()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches and reset states
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process prompt "The capital of France is"
    prompt_tokens = [761, 6512, 315, 9339, 370]
    
    println("Prompt tokens: ", prompt_tokens)
    println("Prompt text: ", join([Inferno.Tokenizer.decode(tokenizer, [t]) for t in prompt_tokens], ""))
    
    # Process each token individually and check state
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(prompt_tokens)
        println("\n=== Processing token $pos: $(Inferno.Tokenizer.decode(tokenizer, [tok])) ===")
        x = model.embed[:, tok]
        
        # Check first SSM layer
        ssm = model.layers[1].op
        println("Before layer 1:")
        println("  SSM state norm: ", sqrt(sum(abs2.(ssm.h))))
        println("  Conv state norm: ", sqrt(sum(abs2.(ssm.conv_state))))
        
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
        
        println("After processing:")
        println("  SSM state norm: ", sqrt(sum(abs2.(ssm.h))))
        println("  Conv state norm: ", sqrt(sum(abs2.(ssm.conv_state))))
    end
    
    # Get prediction
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    
    println("\n=== Final prediction ===")
    top_indices = sortperm(logits, rev=true)[1:15]
    for idx in top_indices
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits[idx], digits=3), ")")
    end
    
    # Check if "[Start thinking]" is in the vocabulary
    start_thinking = Inferno.Tokenizer.encode(tokenizer, "[Start thinking]")
    println("\n\"[Start thinking]\" tokens: ", start_thinking)
    for t in start_thinking
        println("  Token $t: \"", Inferno.Tokenizer.decode(tokenizer, [t]), "\" logit: ", round(logits[t], digits=3))
    end
end

compare_with_llamacpp()
