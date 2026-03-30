using Inferno
using LinearAlgebra

function trace_layer_by_layer_detailed()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    
    println("=== Layer-by-Layer Detailed Trace ===")
    println("Input embedding norm: ", round(sqrt(sum(abs2.(x))), digits=3))
    
    for i in 1:model.config.num_hidden_layers
        layer = model.layers[i]
        layer_type = layer.is_ssm ? "SSM" : "Attn"
        
        x_before = copy(x)
        x_normed = layer.in_norm(x)
        
        # Get the op output
        if layer.is_ssm
            ssm = layer.op
            op_out = ssm(x_normed, pos, model.rope, caches[i])
        else
            attn = layer.op
            op_out = attn(x_normed, pos, model.rope, caches[i])
        end
        
        # Residual
        x_after_op = x_before .+ op_out
        
        # Post norm
        x_post = layer.post_norm(x_after_op)
        
        # MLP
        mlp_out = layer.mlp(x_post)
        
        # Final
        x_final = x_after_op .+ mlp_out
        
        println("\n--- Layer $i ($layer_type) ---")
        println("  x_in norm:       ", round(sqrt(sum(abs2.(x_before))), digits=3))
        println("  x_normed norm:   ", round(sqrt(sum(abs2.(x_normed))), digits=3))
        println("  op_out norm:     ", round(sqrt(sum(abs2.(op_out))), digits=3))
        println("  x_after_op norm: ", round(sqrt(sum(abs2.(x_after_op))), digits=3))
        println("  x_post norm:     ", round(sqrt(sum(abs2.(x_post))), digits=3))
        println("  mlp_out norm:    ", round(sqrt(sum(abs2.(mlp_out))), digits=3))
        println("  x_final norm:    ", round(sqrt(sum(abs2.(x_final))), digits=3))
        
        x = x_final
    end
    
    println("\n=== Final output norm: ", round(sqrt(sum(abs2.(x))), digits=3))
end

trace_layer_by_layer_detailed()
