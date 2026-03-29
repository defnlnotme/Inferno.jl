using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Process single token "The" (760)
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Get embedding
    x = copy(view(model.embed, :, 760))
    println("Embedding: mean=$(mean(x)), std=$(std(x))")

    # Process layer 1 (SSM)
    layer1 = model.layers[1]
    println("\n=== Layer 1 (SSM) ===")

    # In norm
    h = layer1.in_norm(x)
    println("After in_norm: mean=$(mean(h)), std=$(std(h))")

    # SSM projections
    ssm = layer1.op
    qkv = ssm.in_proj * h
    z = ssm.gate_proj * h
    println("QKV: mean=$(mean(qkv)), std=$(std(qkv))")
    println("Z: mean=$(mean(z)), std=$(std(z))")

    # Conv
    println("\nConv state before: ", ssm.conv_state[:, end])

    # Process through SSM
    ModelCPU.reset_states_cpu!(model)
    out1 = layer1.op(h, 0, model.rope, caches[1])
    println("\nAfter SSM: mean=$(mean(out1)), std=$(std(out1))")

    # Residual
    x = x + out1
    println("After residual: mean=$(mean(x)), std=$(std(x))")

    # Post norm
    h = layer1.post_norm(x)
    println("After post_norm: mean=$(mean(h)), std=$(std(h))")

    # MLP
    mlp_out = layer1.mlp(h)
    println("MLP output: mean=$(mean(mlp_out)), std=$(std(mlp_out))")

    # Final residual
    x = x + mlp_out
    println("\nAfter layer 1 complete: mean=$(mean(x)), std=$(std(x))")
end

main()
