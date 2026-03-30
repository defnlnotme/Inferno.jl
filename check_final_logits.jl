using Inferno
using LinearAlgebra

function check_logits()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process prompt
    prompt_tokens = [761, 6512, 315, 9339, 370]
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(prompt_tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    
    # The last hidden state x is now the output after processing all 5 tokens
    # Position for the next token should be 5 (after tokens 0,1,2,3,4)
    
    # Apply final norm
    x_normed = model.final_norm(x)
    
    println("=== Hidden state ===")
    println("Before final norm: ", sqrt(sum(abs2.(x))))
    println("After final norm: ", sqrt(sum(abs2.(x_normed))))
    
    # Get logits
    logits = model.lm_head * x_normed
    
    println("\n=== Logits ===")
    println("Max: ", maximum(logits), " at token ", argmax(logits) - 1)
    println("Min: ", minimum(logits))
    
    # Check specific tokens
    test_tokens = [
        (59, "["),
        (3368, "Start"),
        (7048, " thinking"),
        (61, "]"),
        (272, "\n\n"),
        (221, " "),
        (14, "."),
    ]
    
    println("\n=== Specific token logits ===")
    for (tok, name) in test_tokens
        println("Token $tok (\"$name\"): ", round(logits[tok+1], digits=3))
    end
    
    # Top 10
    top_indices = sortperm(logits, rev=true)[1:10]
    println("\n=== Top 10 ===")
    for idx in top_indices
        println("$(idx-1): ", round(logits[idx], digits=3))
    end
end

check_logits()
