using Inferno
using Statistics

function main()
    println("Loading model...")
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    println("Model loaded.")

    println("\n=== Model dimensions ===")
    println("hidden_size: ", model.config.hidden_size)
    println("num_layers: ", model.config.num_hidden_layers)

    println("\n=== Layer 0 SSM dimensions ===")
    ssm = model.layers[1].op
    println("d_inner: ", ssm.d_inner)
    println("num_v_heads: ", ssm.num_v_heads)
    println("num_k_heads: ", ssm.num_k_heads)
    println("head_k_dim: ", ssm.head_k_dim)
    println("head_v_dim: ", ssm.head_v_dim)
    println("conv_channels: ", ssm.conv_channels)
    println("conv_kernel: ", ssm.conv_kernel)

    println("\n=== Weight shapes ===")
    println("in_proj: ", size(ssm.in_proj))
    println("gate_proj: ", size(ssm.gate_proj))
    println("ssm_conv1d: ", size(ssm.ssm_conv1d))
    println("ssm_out: ", size(ssm.ssm_out))
    println("ssm_alpha_weight: ", size(ssm.ssm_alpha_weight))
    println("ssm_a: ", size(ssm.ssm_a))

    println("\n=== Test forward pass ===")

    # Initialize
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Full forward pass with single token
    h = copy(view(model.embed, :, 1))
    for (i, layer) in enumerate(model.layers)
        h = layer(h, 0, model.rope, caches[i])
    end
    h = model.final_norm(h)
    logits = model.lm_head * h

    println("logits shape: ", size(logits))
    println("any NaN: ", any(isnan, logits))
    println("mean: ", mean(logits))
    println("std: ", std(logits))
    println("min: ", minimum(logits))
    println("max: ", maximum(logits))
    top5_idx = sortperm(logits, rev=true)[1:5]
    println("top 5 tokens: ", top5_idx)
    println("top 5 values: ", logits[top5_idx])

    # Check tokens
    tokens = file.metadata["tokenizer.ggml.tokens"]
    println("\nTop token meanings:")
    for t in top5_idx
        println("  token $t: ", repr(tokens[t + 1]))
    end
end

main()
