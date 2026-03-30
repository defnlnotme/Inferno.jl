using Inferno
using LinearAlgebra

function trace_newline_processing()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches and reset states
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
    
    # Get prediction
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    top_idx = sortperm(logits, rev=true)[1]
    println("After prompt: top prediction = $top_idx")
    
    # Process newline token
    println("\n=== Processing newline token (272) ===")
    x2 = model.embed[:, 272]
    pos2 = 5
    
    for (i, layer) in enumerate(model.layers)
        if layer.is_ssm
            println("\n--- Layer $i (SSM) ---")
            ssm = layer.op
            state_before = copy(ssm.h)
            x2 = layer(x2, pos2, model.rope, caches[i])
            
            # Check state changes
            state_delta = ssm.h - state_before
            println("State delta norm: ", sqrt(sum(abs2.(state_delta))))
            println("State norm after: ", sqrt(sum(abs2.(ssm.h))))
        else
            x2 = layer(x2, pos2, model.rope, caches[i])
        end
    end
    
    x2_normed = model.final_norm(x2)
    logits2 = model.lm_head * x2_normed
    
    println("\n=== After first newline ===")
    println("Logits max: ", maximum(logits2))
    top_indices = sortperm(logits2, rev=true)[1:10]
    for idx in top_indices
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits2[idx], digits=3), ")")
    end
    
    # Check if the model is correctly conditioning
    println("\n=== Checking conditioning ===")
    # Compare the hidden state norms
    println("Hidden state norm after prompt: ", sqrt(sum(abs2.(x))))
    println("Hidden state norm after newline: ", sqrt(sum(abs2.(x2))))
end

trace_newline_processing()
