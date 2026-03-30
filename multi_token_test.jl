using Inferno
using LinearAlgebra

function multi_token_test()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches and reset states
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Prompt
    prompt_tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    # Process all prompt tokens
    println("=== Processing prompt ===")
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(prompt_tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    
    # Get prediction
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    top_idx = sortperm(logits, rev=true)[1]
    println("After prompt, top prediction: $top_idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [top_idx])), "\" (logit: ", round(logits[top_idx], digits=3), ")")
    
    # Generate 10 tokens
    println("\n=== Generating tokens ===")
    generated = Int[]
    x_current = x
    
    for gen_pos in 1:10
        pred_token = top_idx
        push!(generated, pred_token)
        
        # Process predicted token
        x_next = model.embed[:, pred_token]
        pos = length(prompt_tokens) + gen_pos - 1
        
        for (i, layer) in enumerate(model.layers)
            x_next = layer(x_next, pos, model.rope, caches[i])
        end
        
        # Get next prediction
        x_normed = model.final_norm(x_next)
        logits = model.lm_head * x_normed
        top_idx = sortperm(logits, rev=true)[1]
        
        println("Generated token $gen_pos: $pred_token -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [pred_token])), "\"")
        println("  Next top prediction: $top_idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [top_idx])), "\" (logit: ", round(logits[top_idx], digits=3), ")")
    end
    
    println("\n=== Generated text ===")
    full_output = Inferno.Tokenizer.decode(tokenizer, generated)
    println(full_output)
end

multi_token_test()
