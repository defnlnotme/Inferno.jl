using Inferno
using Inferno.ModelCPU
using Inferno.Tokenizer

function main()
    # Load the model
    model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Tokenize a simple prompt
    prompt = "2 + 2 ="
    tokens = encode(tok, prompt)
    println("Tokens: ", tokens)
    println("Prompt decoded: \"", decode(tok, tokens), "\"")

    # Initialize cache
    cache_size = model.config.max_position_embeddings
    caches = [init_kv_cache_cpu(model.config, cache_size) for _ in 1:model.config.num_hidden_layers]

    # Process prompt tokens
    println("\nProcessing first ", length(tokens)-1, " tokens...")
    logits = forward_cpu!(model, tokens[1:end-1], 0, caches)
    println("Logits shape: ", size(logits))
    
    # Check logits for the last position
    last_logits = logits[:, end]
    println("\nLogits at last position:")
    println("  min: ", minimum(last_logits))
    println("  max: ", maximum(last_logits))
    
    # Top 10 tokens
    top10 = sortperm(last_logits, rev=true)[1:10]
    println("\nTop 10 tokens at last position:")
    for (i, idx) in enumerate(top10)
        println("  $i. Token $idx: \"", tok.id_to_token[idx+1], "\" (logit: ", last_logits[idx], ")")
    end
    
    curr_pos = length(tokens) - 1

    # Generate from last prompt token
    println("\nGenerating from last prompt token...")
    next_tok = tokens[end]
    logits2 = forward_cpu!(model, [next_tok], curr_pos, caches)
    curr_pos += 1
    
    println("\nLogits after generating from last prompt token:")
    println("  min: ", minimum(logits2))
    println("  max: ", maximum(logits2))
    
    top10_2 = sortperm(logits2[:, end], rev=true)[1:10]
    println("\nTop 10 tokens:")
    for (i, idx) in enumerate(top10_2)
        println("  $i. Token $idx: \"", tok.id_to_token[idx+1], "\" (logit: ", logits2[idx, end], ")")
    end
    
    next_tok = argmax(logits2[:, end])
    println("\nGreedy selection: Token $next_tok = \"", tok.id_to_token[next_tok+1], "\"")
end

main()
