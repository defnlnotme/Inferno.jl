using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Use correct tokens: "The capital of France is" -> [760, 6511, 314, 9338, 369]
    prompt_tokens = [760, 6511, 314, 9338, 369]

    println("Testing token-by-token:")
    
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Process first token only
    pos = 0
    tok_id = prompt_tokens[1]
    h = copy(view(model.embed, :, tok_id))
    
    println("After embed: mean=$(mean(h)), std=$(std(h))")

    # Process first layer (SSM)
    layer1 = model.layers[1]
    h1 = layer1(h, pos, model.rope, caches[1])
    
    println("After layer 1 (SSM): mean=$(mean(h1)), std=$(std(h1))")
    println("  h1[1:5]: ", h1[1:5])

    # Process all layers
    h = copy(view(model.embed, :, tok_id))
    for (i, layer) in enumerate(model.layers)
        h = layer(h, pos, model.rope, caches[i])
    end
    
    println("\nAfter all layers: mean=$(mean(h)), std=$(std(h))")
    
    h_normed = model.final_norm(h)
    println("After final_norm: mean=$(mean(h_normed)), std=$(std(h_normed))")
    
    logits = model.lm_head * h_normed
    println("\nLogits: mean=$(mean(logits)), std=$(std(logits))")
    
    top5 = sortperm(logits, rev=true)[1:5]
    println("Top 5 tokens:")
    for t in top5
        println("  Token $t: ", repr(tokens[t+1]), " (logit=", logits[t], ")")
    end
end

main()
