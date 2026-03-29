using Inferno
using LinearAlgebra

function trace_prompt()
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Prompt tokens
    tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    # Process first N-1 tokens (prefill)
    println("=== Processing prompt tokens [1:end-1] ===")
    for (pos, tok) in enumerate(tokens[1:end-1])
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
        println("After token $pos ($(Inferno.Tokenizer.decode(tokenizer, [tok]))): x norm = ", sqrt(sum(abs2.(x))))
    end
    
    # Now process the last token and get prediction
    println("\n=== Processing last prompt token ===")
    last_tok = tokens[end]
    x = model.embed[:, last_tok]
    pos = length(tokens) - 1  # Position 4
    
    for (i, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, caches[i])
    end
    
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    
    top_indices = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 predictions after prompt:")
    for idx in top_indices
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits[idx], digits=3), ")")
    end
    
    # Now process the predicted token
    println("\n=== Processing first generated token ===")
    pred_tok = top_indices[1]
    x2 = model.embed[:, pred_tok]
    pos2 = pos + 1
    
    for (i, layer) in enumerate(model.layers)
        x2 = layer(x2, pos2, model.rope, caches[i])
    end
    
    x2_normed = model.final_norm(x2)
    logits2 = model.lm_head * x2_normed
    
    top_indices2 = sortperm(logits2, rev=true)[1:10]
    println("\nTop 10 predictions after generated token:")
    for idx in top_indices2
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits2[idx], digits=3), ")")
    end
end

trace_prompt()
