using Inferno
using LinearAlgebra

function trace_first_token_layer7()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761  # "The"
    pos = 0
    
    # Process through layers 1-6
    x = model.embed[:, tok]
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    println("=== Input to layer 7 (first token) ===")
    println("Norm: ", round(sqrt(sum(abs2.(x))), digits=3))
    
    # Process layer 7
    layer = model.layers[7]
    x_orig = copy(x)
    
    # Step by step
    x_norm = layer.in_norm(x)
    println("\nAfter in_norm: ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    
    ssm_out = layer.op(x_norm, pos, model.rope, caches[7])
    println("SSM output: ", round(sqrt(sum(abs2.(ssm_out))), digits=3))
    
    x_after_ssm = x_orig .+ ssm_out
    println("After residual: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    
    x_norm2 = layer.post_norm(x_after_ssm)
    println("After post_norm: ", round(sqrt(sum(abs2.(x_norm2))), digits=3))
    
    mlp_out = layer.mlp(x_norm2)
    println("MLP output: ", round(sqrt(sum(abs2.(mlp_out))), digits=3))
    
    x_final = x_after_ssm .+ mlp_out
    println("\nFinal: ", round(sqrt(sum(abs2.(x_final))), digits=3))
    
    # Now process second token
    println("\n=== Second token (6512) ===")
    tok2 = 6512
    pos2 = 1
    
    x2 = model.embed[:, tok2]
    for i in 1:6
        x2 = model.layers[i](x2, pos2, model.rope, caches[i])
    end
    
    println("Input to layer 7: ", round(sqrt(sum(abs2.(x2))), digits=3))
    
    x2_orig = copy(x2)
    x2_norm = layer.in_norm(x2)
    println("After in_norm: ", round(sqrt(sum(abs2.(x2_norm))), digits=3))
    
    ssm_out2 = layer.op(x2_norm, pos2, model.rope, caches[7])
    println("SSM output: ", round(sqrt(sum(abs2.(ssm_out2))), digits=3))
    
    x2_after_ssm = x2_orig .+ ssm_out2
    println("After residual: ", round(sqrt(sum(abs2.(x2_after_ssm))), digits=3))
end

trace_first_token_layer7()
