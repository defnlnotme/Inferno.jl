using Inferno
using LinearAlgebra

function trace_inference()
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process a single token
    tokens = [761]  # "The"
    tok = tokens[1]
    pos = 0
    
    # Get embedding
    x = model.embed[:, tok]
    println("Initial embedding norm: ", sqrt(sum(abs2.(x))))
    
    # Process through layers
    for (i, layer) in enumerate(model.layers)
        x_before = copy(x)
        x = layer(x, pos, model.rope, caches[i])
        
        # Check for NaN/Inf
        if any(isnan.(x)) || any(isinf.(x))
            println("ERROR: NaN/Inf after layer $i")
            break
        end
        
        # Check norm change
        delta = x - x_before
        println("Layer $i ($(layer.is_ssm ? "SSM" : "Attn")): x norm = $(sqrt(sum(abs2.(x)))), delta norm = $(sqrt(sum(abs2.(delta))))")
    end
    
    # Final norm and logits
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    
    println("\nFinal hidden norm: ", sqrt(sum(abs2.(x))))
    println("After final norm: ", sqrt(sum(abs2.(x_normed))))
    println("Logits max: ", maximum(logits))
    println("Logits min: ", minimum(logits))
    
    # Top predictions
    top_indices = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 predictions:")
    for idx in top_indices
        println("  $idx -> \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [idx])), "\" (logit: ", round(logits[idx], digits=3), ")")
    end
end

trace_inference()
