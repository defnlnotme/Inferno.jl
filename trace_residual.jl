using Inferno
using LinearAlgebra

function trace_residual_growth()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token "The"
    x = model.embed[:, 761]
    pos = 0
    
    println("=== Initial state ===")
    println("Input norm: ", sqrt(sum(abs2.(x))))
    
    # Process through first layer manually
    layer = model.layers[1]
    x_orig = copy(x)
    
    # Step 1: Input norm
    x_norm = layer.in_norm(x)
    println("\nAfter in_norm: ", sqrt(sum(abs2.(x_norm))))
    
    # Step 2: SSM
    ssm_out = layer.op(x_norm, pos, model.rope, caches[1])
    println("SSM output norm: ", sqrt(sum(abs2.(ssm_out))))
    
    # Step 3: First residual
    x_after_ssm = x_orig + ssm_out
    println("After SSM residual: ", sqrt(sum(abs2.(x_after_ssm))))
    
    # Step 4: Post norm
    x_norm2 = layer.post_norm(x_after_ssm)
    println("After post_norm: ", sqrt(sum(abs2.(x_norm2))))
    
    # Step 5: MLP
    mlp_out = layer.mlp(x_norm2)
    println("MLP output norm: ", sqrt(sum(abs2.(mlp_out))))
    
    # Step 6: Final residual
    x_final = x_after_ssm + mlp_out
    println("After MLP residual: ", sqrt(sum(abs2.(x_final))))
    
    # Compare with layer call
    x_layer = layer(x_orig, pos, model.rope, caches[1])
    println("\nDirect layer call: ", sqrt(sum(abs2.(x_layer))))
    
    println("\n=== Analysis ===")
    println("SSM output relative to input: ", sqrt(sum(abs2.(ssm_out))) / sqrt(sum(abs2.(x_orig))))
    println("MLP output relative to post-SSM: ", sqrt(sum(abs2.(mlp_out))) / sqrt(sum(abs2.(x_after_ssm))))
end

trace_residual_growth()
