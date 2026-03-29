using Inferno
using LinearAlgebra

function trace_generation()
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Prompt tokens
    tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    # Process all prompt tokens
    println("=== Processing prompt ===")
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    
    # Get first prediction
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    pred1 = sortperm(logits, rev=true)[1]
    println("First prediction: $pred1 -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [pred1])), "\"")
    
    # Process the newline token
    println("\n=== Processing token 272 (\\n\\n) ===")
    x2 = model.embed[:, 272]
    pos2 = length(tokens)
    
    for (i, layer) in enumerate(model.layers)
        x2 = layer(x2, pos2, model.rope, caches[i])
    end
    
    x2_normed = model.final_norm(x2)
    logits2 = model.lm_head * x2_normed
    
    top_indices2 = sortperm(logits2, rev=true)[1:10]
    println("Top 10 predictions after \\n\\n:")
    for idx in top_indices2
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits2[idx], digits=3), ")")
    end
end

trace_generation()
