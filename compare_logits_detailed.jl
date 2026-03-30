using Inferno
using LinearAlgebra

function compare_logits_detailed()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches and reset states
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process prompt "The capital of France is"
    prompt_tokens = [761, 6512, 315, 9339, 370]
    
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(prompt_tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    
    # Get logits
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    
    # Output specific logits for comparison
    println("=== Logits for specific tokens ===")
    test_tokens = [
        (272, "\\n\\n"),
        (221, " "),
        (14, "."),
        (59, "["),
        (3368, "Start"),
        (7048, " thinking"),
        (61, "]"),
        (151645, "<|endoftext|>"),
        (151644, "<|im_start|>"),
    ]
    
    for (tok, name) in test_tokens
        println("Token $tok (\"$name\"): $(round(logits[tok+1], digits=4))")
    end
    
    # Top 10 tokens
    println("\n=== Top 10 predictions ===")
    top_indices = sortperm(logits, rev=true)[1:10]
    for idx in top_indices
        println("$(idx-1) -> \"$(escape_string(Inferno.Tokenizer.decode(tokenizer, [idx-1])))\": $(round(logits[idx], digits=4))")
    end
end

compare_logits_detailed()
