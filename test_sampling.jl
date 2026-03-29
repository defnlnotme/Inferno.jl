using Inferno
using LinearAlgebra

function test_sampling()
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Prompt tokens
    tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    # Process all prompt tokens
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    
    # Get logits for first prediction
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    
    # Check what "Paris" token would have
    paris_tokens = Inferno.Tokenizer.encode(tokenizer, "Paris")
    println("Paris tokens: ", paris_tokens)
    
    for pt in paris_tokens
        if pt <= length(logits)
            println("  Logit for token $pt (\"$(Inferno.Tokenizer.decode(tokenizer, [pt]))\"): ", logits[pt])
        end
    end
    
    # Compare with top prediction
    top_idx = sortperm(logits, rev=true)[1]
    println("\nTop prediction: $top_idx (\"$(Inferno.Tokenizer.decode(tokenizer, [top_idx]))\") with logit ", logits[top_idx])
    
    # Check logits for space + Paris
    space_paris = Inferno.Tokenizer.encode(tokenizer, " Paris")
    println("\n\" Paris\" tokens: ", space_paris)
    for pt in space_paris
        if pt <= length(logits)
            println("  Logit for token $pt (\"$(Inferno.Tokenizer.decode(tokenizer, [pt]))\"): ", logits[pt])
        end
    end
end

test_sampling()
