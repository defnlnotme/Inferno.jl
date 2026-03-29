using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

    h = copy(model.embed[:, 761])  # Token "The"
    println("Input norm: ", round(sqrt(sum(abs2, h)), digits=3))

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    for (i, layer) in enumerate(model.layers)
        h = layer(h, 0, model.rope, caches[i])
        println("After layer $i ($(layer.is_ssm ? "SSM" : "Attn")): norm=$(round(sqrt(sum(abs2, h)), digits=3))")
    end

    # Final norm
    h_normed = model.final_norm(h)
    println("\nAfter final_norm: norm=$(round(sqrt(sum(abs2, h_normed)), digits=3))")

    # LM head
    logits = model.lm_head * h_normed
    println("Logits: mean=$(round(mean(logits), digits=3)), max=$(round(maximum(logits), digits=2))")

    # Top 5
    top5 = sortperm(logits, rev=true)[1:5]
    tokens = file.metadata["tokenizer.ggml.tokens"]
    println("\nTop 5:")
    for t in top5
        println("  $(repr(tokens[t+1])): $(round(logits[t], digits=3))")
    end
end

main()
