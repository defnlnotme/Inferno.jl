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

    # Get first token embedding
    first_tok = tokens[1]
    println("\nFirst token: ", first_tok)
    
    # Get embedding
    x = model.embed[:, first_tok]
    println("Embedding shape: ", size(x))
    println("Embedding norm: ", sum(abs2, x))
    println("Embedding sample: ", x[1:5])

    # Initialize cache
    cache_size = model.config.max_position_embeddings
    caches = [init_kv_cache_cpu(model.config, cache_size) for _ in 1:model.config.num_hidden_layers]

    # Process through each layer
    println("\nProcessing through layers:")
    pos = 0
    for (i, layer) in enumerate(model.layers)
        x_before = copy(x)
        x = layer(x, pos, model.rope, caches[i])
        x_diff = sum(abs2, x - x_before)
        println("  Layer $i ($(layer.is_ssm ? "SSM" : "Attention")): norm=$(round(sum(abs2, x), digits=2)), diff=$(round(x_diff, digits=2))")
    end

    # Final norm
    println("\nFinal norm:")
    x_before = copy(x)
    x = model.final_norm(x)
    x_diff = sum(abs2, x - x_before)
    println("  norm=$(round(sum(abs2, x), digits=2)), diff=$(round(x_diff, digits=2))")

    # LM head
    println("\nLM head:")
    println("  LM head shape: ", size(model.lm_head))
    logits = model.lm_head * x
    println("  Logits shape: ", size(logits))
    println("  Logits min: ", minimum(logits))
    println("  Logits max: ", maximum(logits))

    # Top 10 tokens
    top10 = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 tokens:")
    for (i, idx) in enumerate(top10)
        println("  $i. Token $idx: \"", tok.id_to_token[idx+1], "\" (logit: ", round(logits[idx], digits=2), ")")
    end
end

main()
