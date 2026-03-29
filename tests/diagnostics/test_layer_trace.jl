using Inferno
using Statistics

function main()
    # Test with F16 model
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Get embedding for "The"
    h = copy(model.embed[:, 761])  # Token "The" (1-indexed)
    println("Input embedding norm: ", sqrt(sum(abs2, h)))
    println("Input embedding mean: ", mean(h))
    println("Input embedding std: ", std(h))

    # Forward through first layer only
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    layer = model.layers[1]
    println("\nLayer 1 is SSM: ", layer.is_ssm)

    # Apply in_norm
    h_norm = layer.in_norm(h)
    println("\nAfter in_norm:")
    println("  norm: ", sqrt(sum(abs2, h_norm)))
    println("  mean: ", mean(h_norm))
    println("  std: ", std(h_norm))

    # Forward through the layer
    h_out = layer(h, 0, model.rope, caches[1])
    
    println("\nAfter layer 1:")
    println("  norm: ", sqrt(sum(abs2, h_out)))
    println("  mean: ", mean(h_out))
    println("  std: ", std(h_out))
end

main()
