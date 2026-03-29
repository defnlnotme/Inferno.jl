using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

    h = copy(model.embed[:, 761])
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Forward through layer 1
    layer = model.layers[1]
    ssm = layer.op
    
    # Apply in_norm
    h_norm = layer.in_norm(h)
    
    # QKV projection
    qkv = ssm.in_proj * h_norm
    
    # Check conv weights
    println("Conv kernel shape: ", size(ssm.ssm_conv1d))
    println("Conv kernel stats:")
    println("  mean: ", round(mean(ssm.ssm_conv1d), digits=5))
    println("  std: ", round(std(ssm.ssm_conv1d), digits=5))
    println("  First row (for first channel): ", ssm.ssm_conv1d[:, 1])
    
    # Apply convolution with fresh state
    conv_out = zeros(Float32, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        conv_out[c] = ssm.ssm_conv1d[4, c] * qkv[c]  # Only use the current position
    end
    
    println("\nConv output (single position):")
    println("  norm: ", round(sqrt(sum(abs2, conv_out)), digits=3))
    
    # Compare with expected
    # For a 1D causal conv with kernel size 4, the first position should only use the first element
    # But since conv_state is all zeros, the output should be conv1d[4, :] * qkv
    println("\nExpected: conv1d[4, :] * qkv")
    println("  norm: ", round(sqrt(sum(abs2, ssm.ssm_conv1d[4, :] .* qkv)), digits=3))
end

main()
