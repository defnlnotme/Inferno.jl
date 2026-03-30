using Inferno
using LinearAlgebra

function analyze_direction()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    
    # Process through first 5 layers
    for i in 1:5
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    layer = model.layers[6]
    
    # Process SSM
    x_before = copy(x)
    x_norm = layer.in_norm(x)
    ssm_output = layer.op(x_norm, pos, model.rope, caches[6])
    x_after_ssm = x_before .+ ssm_output
    
    # Post norm
    x_post = layer.post_norm(x_after_ssm)
    
    # MLP
    mlp = layer.mlp
    gate = mlp.gate_weight * x_post
    gate_silu = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
    up = mlp.up_weight * x_post
    hidden = gate_silu .* up
    mlp_out = mlp.down_weight * hidden
    
    # Final output
    x_final = x_after_ssm .+ mlp_out
    
    println("=== Direction Analysis ===")
    println("\nResidual (x_after_ssm):")
    println("  norm: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    println("  first 5 elements: ", round.(x_after_ssm[1:5], digits=4))
    
    println("\nMLP output:")
    println("  norm: ", round(sqrt(sum(abs2.(mlp_out))), digits=3))
    println("  first 5 elements: ", round.(mlp_out[1:5], digits=4))
    
    println("\nFinal output:")
    println("  norm: ", round(sqrt(sum(abs2.(x_final))), digits=3))
    println("  first 5 elements: ", round.(x_final[1:5], digits=4))
    
    # Compute angle
    dot_product = dot(x_after_ssm, mlp_out)
    angle = acos(dot_product / (sqrt(sum(abs2.(x_after_ssm))) * sqrt(sum(abs2.(mlp_out)))))
    
    println("\nAngle between residual and MLP output:")
    println("  dot product: ", round(dot_product, digits=3))
    println("  angle: ", round(rad2deg(angle), digits=1), " degrees")
    
    # Check if the post-norm is correct
    println("\n=== Post-Norm Analysis ===")
    println("Input to post_norm (x_after_ssm):")
    println("  norm: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    
    println("\nPost-norm weight:")
    println("  mean: ", round(sum(layer.post_norm.weight) / length(layer.post_norm.weight), digits=3))
    println("  first 5: ", round.(layer.post_norm.weight[1:5], digits=3))
    
    println("\nOutput of post_norm (x_post):")
    println("  norm: ", round(sqrt(sum(abs2.(x_post))), digits=3))
    println("  first 5: ", round.(x_post[1:5], digits=4))
    
    # Check if x_post is orthogonal to x_after_ssm
    dot_x_post_x_after_ssm = dot(x_post, x_after_ssm)
    println("\nOrthogonality check:")
    println("  dot(x_post, x_after_ssm) = ", round(dot_x_post_x_after_ssm, digits=3))
    
    # The key insight: RMS norm preserves direction!
    # x_post = x_after_ssm / sqrt(mean(x_after_ssm^2) + eps) * weight
    # So x_post should be parallel to x_after_ssm
    
    # Check if they're parallel
    x_after_ssm_normed = x_after_ssm ./ sqrt(sum(abs2.(x_after_ssm)))
    x_post_normed = x_post ./ sqrt(sum(abs2.(x_post)))
    
    dot_normalized = dot(x_after_ssm_normed, x_post_normed)
    println("  dot(normalized): ", round(dot_normalized, digits=4))
    
    if abs(dot_normalized - 1.0) < 0.01
        println("  -> x_post and x_after_ssm are PARALLEL (expected for RMS norm)")
    else
        println("  -> x_post and x_after_ssm are NOT parallel (unexpected!)")
    end
end

analyze_direction()
