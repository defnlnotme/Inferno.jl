using Inferno
using LinearAlgebra

function debug_layer7()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process tokens up to layer 6
    prompt_tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    x = zeros(Float32, model.config.hidden_size)
    
    for (pos, tok) in enumerate(prompt_tokens)
        x = model.embed[:, tok]
        for i in 1:6
            x = model.layers[i](x, pos-1, model.rope, caches[i])
        end
    end
    
    println("=== Input to Layer 7 ===")
    println("Norm: ", sqrt(sum(abs2.(x))))
    println("Sample: ", x[1:5])
    
    # Process layer 7 manually
    layer = model.layers[7]
    pos = 4  # Last position
    
    x_orig = copy(x)
    
    # Step 1: Input norm
    x_norm = layer.in_norm(x)
    println("\nAfter in_norm: ", sqrt(sum(abs2.(x_norm))))
    
    # Step 2: SSM - check what it's doing
    ssm = layer.op
    println("\n=== SSM details ===")
    println("SSM h state norm before: ", sqrt(sum(abs2.(ssm.h))))
    
    ssm_out = layer.op(x_norm, pos, model.rope, caches[7])
    
    println("SSM h state norm after: ", sqrt(sum(abs2.(ssm.h))))
    println("SSM output norm: ", sqrt(sum(abs2.(ssm_out))))
    println("SSM output sample: ", ssm_out[1:5])
    
    # Check SSM parameters
    println("\nSSM A sample: ", ssm.A[1:5])
    println("SSM A range: [", minimum(ssm.A), ", ", maximum(ssm.A), "]")
    
    # Step 3: Residual
    x_after_ssm = x_orig .+ ssm_out
    println("\nAfter residual: ", sqrt(sum(abs2.(x_after_ssm))))
    
    # Step 4: Post norm
    x_norm2 = layer.post_norm(x_after_ssm)
    println("After post_norm: ", sqrt(sum(abs2.(x_norm2))))
    
    # Step 5: MLP
    mlp_out = layer.mlp(x_norm2)
    println("\nMLP output norm: ", sqrt(sum(abs2.(mlp_out))))
    
    # Final
    x_final = x_after_ssm .+ mlp_out
    println("\nFinal layer 7 output: ", sqrt(sum(abs2.(x_final))))
end

debug_layer7()
