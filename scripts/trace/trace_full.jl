using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tok = file.metadata["tokenizer.ggml.tokens"]

    # Process a single token
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    x = copy(view(model.embed, :, 760))  # "The"

    println("Initial embedding: mean=$(mean(x)), std=$(std(x))")

    # Process all 24 layers
    for (i, layer) in enumerate(model.layers)
        h_before = copy(x)
        x = layer(x, 0, model.rope, caches[i])
        diff = x - h_before
        println("Layer $i: output mean=$(mean(x)), std=$(std(x)), delta mean=$(mean(diff)), delta std=$(std(diff))")
    end

    # Final norm
    h_normed = model.final_norm(x)
    println("\nAfter final_norm: mean=$(mean(h_normed)), std=$(std(h_normed))")

    # Logits
    logits = model.lm_head * h_normed
    println("Logits: mean=$(mean(logits)), std=$(std(logits)), min=$(minimum(logits)), max=$(maximum(logits))")

    # Top tokens
    top10 = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 tokens:")
    for t in top10
        println("  Token $t: ", repr(tok[t+1]), " (logit=", logits[t], ")")
    end
end

main()
