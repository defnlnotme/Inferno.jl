using Inferno
using LinearAlgebra

function trace_all_layers()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    x = model.embed[:, 761]  # "The"
    pos = 0
    
    println("=== Layer-by-layer norm progression ===")
    println("Layer | Type | InNorm | OpOut | Res1 | PostNorm | MLP | Final")
    println("------|------|--------|-------|------|----------|-----|------")
    
    for (i, layer) in enumerate(model.layers)
        x_orig = copy(x)
        
        # Step 1: Input norm
        x_norm = layer.in_norm(x)
        in_norm_val = sqrt(sum(abs2.(x_norm)))
        
        # Step 2: Op (SSM or Attention)
        op_out = layer.op(x_norm, pos, model.rope, caches[i])
        op_norm = sqrt(sum(abs2.(op_out)))
        
        # Step 3: First residual
        x_after_op = x_orig .+ op_out
        res1_norm = sqrt(sum(abs2.(x_after_op)))
        
        # Step 4: Post norm
        x_norm2 = layer.post_norm(x_after_op)
        post_norm_val = sqrt(sum(abs2.(x_norm2)))
        
        # Step 5: MLP
        mlp_out = layer.mlp(x_norm2)
        mlp_norm = sqrt(sum(abs2.(mlp_out)))
        
        # Step 6: Final residual
        x_final = x_after_op .+ mlp_out
        final_norm = sqrt(sum(abs2.(x_final)))
        
        layer_type = layer.is_ssm ? "SSM" : "Attn"
        println("$i | $layer_type | $(round(in_norm_val, digits=1)) | $(round(op_norm, digits=2)) | $(round(res1_norm, digits=2)) | $(round(post_norm_val, digits=1)) | $(round(mlp_norm, digits=2)) | $(round(final_norm, digits=2))")
        
        # Update x for next layer
        x = x_final
    end
    
    println("\nFinal hidden norm: ", sqrt(sum(abs2.(x))))
end

trace_all_layers()
