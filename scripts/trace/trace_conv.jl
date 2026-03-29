using Inferno
using Statistics

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Get embedding for token 760 ("The")
    x = copy(view(model.embed, :, 760))

    # Apply in_norm
    layer1 = model.layers[1]
    h = layer1.in_norm(x)

    # SSM projections
    ssm = layer1.op
    qkv = ssm.in_proj * h

    println("h (after in_norm): mean=$(mean(h)), std=$(std(h))")
    println("qkv: mean=$(mean(qkv)), std=$(std(qkv))")

    # At position 0, after setting conv_state[:, 4] = qkv
    # x_conv = sum over k of conv_state[:, k] * ssm_conv1d[k, :]
    # At pos 0: x_conv = qkv * ssm_conv1d[4, :] (since conv_state[:, 1:3] = 0)

    x_conv_manual = qkv .* ssm.ssm_conv1d[4, :]
    println("\nManual x_conv at pos 0: mean=$(mean(x_conv_manual)), std=$(std(x_conv_manual))")

    # Apply SiLU
    x_conv_silu = x_conv_manual .* (1.0f0 ./ (1.0f0 .+ exp.(-x_conv_manual)))
    println("After SiLU: mean=$(mean(x_conv_silu)), std=$(std(x_conv_silu))")

    # Check if this matches what the SSM actually computes
    ModelCPU.reset_states_cpu!(model)
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]

    out = ssm(h, 0, model.rope, caches[1])
    println("\nActual SSM output: mean=$(mean(out)), std=$(std(out))")
end

main()
