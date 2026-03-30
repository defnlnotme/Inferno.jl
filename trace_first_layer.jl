using Inferno
using LinearAlgebra

function trace_first_layer_detailed()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token "The" (token 761)
    tok = 761
    x = model.embed[:, tok]
    pos = 0
    
    println("=== Embedding ===")
    println("Token $tok embedding norm: ", sqrt(sum(abs2.(x))))
    println("Embedding sample values: ", x[1:5])
    
    # Process through first layer manually
    layer = model.layers[1]
    
    println("\n=== First Layer (SSM) ===")
    println("Layer is SSM: ", layer.is_ssm)
    
    # Step 1: Input norm
    x_norm = layer.in_norm(x)
    println("\nAfter in_norm:")
    println("  Norm: ", sqrt(sum(abs2.(x_norm))))
    println("  Sample: ", x_norm[1:5])
    
    # Check in_norm weights
    println("\nIn_norm weight sample: ", layer.in_norm.weight[1:5])
    println("In_norm weight mean: ", sum(layer.in_norm.weight) / length(layer.in_norm.weight))
    
    # Step 2: SSM
    ssm_out = layer.op(x_norm, pos, model.rope, caches[1])
    println("\nSSM output:")
    println("  Norm: ", sqrt(sum(abs2.(ssm_out))))
    println("  Sample: ", ssm_out[1:5])
    
    # Step 3: Residual
    x_after_ssm = x .+ ssm_out
    println("\nAfter SSM residual (x + ssm_out):")
    println("  Norm: ", sqrt(sum(abs2.(x_after_ssm))))
    
    # Step 4: Post norm
    x_norm2 = layer.post_norm(x_after_ssm)
    println("\nAfter post_norm:")
    println("  Norm: ", sqrt(sum(abs2.(x_norm2))))
    println("  Post_norm weight mean: ", sum(layer.post_norm.weight) / length(layer.post_norm.weight))
    
    # Step 5: MLP
    mlp_out = layer.mlp(x_norm2)
    println("\nMLP output:")
    println("  Norm: ", sqrt(sum(abs2.(mlp_out))))
    println("  Sample: ", mlp_out[1:5])
    
    # Step 6: Final residual
    x_final = x_after_ssm .+ mlp_out
    println("\n=== Final layer 1 output ===")
    println("Norm: ", sqrt(sum(abs2.(x_final))))
    println("Sample: ", x_final[1:5])
    
    # Compare with direct layer call
    Inferno.ModelCPU.reset_states_cpu!(model)
    x_direct = layer(model.embed[:, tok], pos, model.rope, caches[1])
    println("\nDirect layer call norm: ", sqrt(sum(abs2.(x_direct))))
end

trace_first_layer_detailed()
