using Inferno
using Statistics

function main()
    cpu_model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Full prompt: "The capital of France is"
    prompt_tokens = [760, 6511, 314, 9338, 369]

    caches = [ModelCPU.init_kv_cache_cpu(cpu_model.config, 512) for _ in 1:cpu_model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(cpu_model)

    h = nothing
    for (pos, tok_id) in enumerate(prompt_tokens)
        h = copy(view(cpu_model.embed, :, tok_id))
        for (i, layer) in enumerate(cpu_model.layers)
            h = layer(h, pos-1, cpu_model.rope, caches[i])
        end
    end

    h_normed = cpu_model.final_norm(h)
    logits = cpu_model.lm_head * h_normed

    println("Logits: mean=", mean(logits), " std=", std(logits))
    println("\nTop 10 tokens:")
    top10 = sortperm(logits, rev=true)[1:10]
    for t in top10
        println("  Token $t: ", repr(tokens[t+1]), " (logit=", logits[t], ")")
    end

    # Check Paris
    println("\n' Paris' (11751): ", logits[11751])
end

main()
