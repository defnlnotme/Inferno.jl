# Add src to LOAD_PATH to find Inferno module
if !("src" in LOAD_PATH)
    push!(LOAD_PATH, "src")
end

using oneAPI
using Inferno

function test_qwen35()
    model_path = "/home/fra/dev/inferno/tests/models/Qwen3.5-0.8B-UD-IQ2_XXS.gguf"
    
    println("--- Testing Qwen3.5 0.8B (Pure Julia + oneAPI) ---")
    
    # Selecting the second GPU per user rule
    devs = collect(oneAPI.devices())
    if length(devs) >= 2
        oneAPI.device!(devs[2])
    end
    println("Using device: ", oneAPI.device())

    # Load model
    model, tok = Inferno.load_model(model_path) # Inferno.jl uses GPU 2 by default if available
    
    println("Generating text...")
    prompt = "Hello, how are you?"
    tokens = [tok.bos_id]
    append!(tokens, Inferno.Tokenizer.encode(tok, prompt))
    
    # Initial states
    caches = [Inferno.Model.init_kv_cache(
        model.config.head_dim, 
        model.config.num_key_value_heads, 
        model.config.max_position_embeddings
    ) for _ in 1:model.config.num_hidden_layers]
    
    # Forward pass
    logits = Inferno.Model.forward!(model, tokens, 0, caches)
    
    println("Logits size: ", size(logits))
    println("Top 5 logits for last token:")
    last_logits = logits[:, end]
    top_indices = sortperm(last_logits, rev=true)[1:5]
    for idx in top_indices
        token_str = Inferno.Tokenizer.decode(tok, [idx-1])
        println("  $token_str: $(last_logits[idx])")
    end
    
    println("--- Test Complete ---")
end

test_qwen35()
