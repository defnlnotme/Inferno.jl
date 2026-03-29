using Inferno
using Statistics

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Process token 760 "The"
    pos = 0
    tok = 760

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    h = copy(model.embed[:, tok])
    println("After embed: mean=$(mean(h)), std=$(std(h))")

    for (i, layer) in enumerate(model.layers)
        h_before = h
        h = layer(h, pos, model.rope, caches[i])
        println("After layer $i ($(layer.is_ssm ? "SSM" : "Attn")): mean=$(mean(h)), std=$(std(h)), max=$(maximum(abs.(h)))")

        # Check for explosion
        if maximum(abs.(h)) > 100
            println("  WARNING: Large values detected!")
            break
        end
    end

    h_normed = model.final_norm(h)
    println("\nAfter final_norm: mean=$(mean(h_normed)), std=$(std(h_normed))")

    logits = model.lm_head * h_normed
    println("Logits: mean=$(mean(logits)), std=$(std(logits)), max=$(maximum(logits)), min=$(minimum(logits))")

    top5 = sortperm(logits, rev=true)[1:5]
    println("\nTop 5 tokens:")
    for t in top5
        println("  Token $t: logit=$(logits[t])")
    end
end

main()
