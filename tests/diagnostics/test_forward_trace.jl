using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Get embedding for "The"
    h = copy(model.embed[:, 761])  # Token "The" (1-indexed)
    println("Input embedding norm: ", sqrt(sum(abs2, h)))

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Forward through all layers
    for (i, layer) in enumerate(model.layers)
        h = layer(h, 0, model.rope, caches[i])
        if i <= 3 || i == length(model.layers)
            println("After layer $i: norm=$(round(sqrt(sum(abs2, h)), digits=2))")
        end
    end

    # Final norm
    h_normed = model.final_norm(h)
    println("\nAfter final_norm: norm=$(round(sqrt(sum(abs2, h_normed)), digits=2))")

    # LM head
    logits = model.lm_head * h_normed
    println("Logits: mean=$(round(mean(logits), digits=3)), std=$(round(std(logits), digits=3)), max=$(round(maximum(logits), digits=2)), min=$(round(minimum(logits), digits=2))")

    # Top tokens
    top10 = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 predictions:")
    for t in top10
        println("  Token $t: $(repr(tokens[t+1])) (logit=$(round(logits[t], digits=3)))")
    end
end

main()
