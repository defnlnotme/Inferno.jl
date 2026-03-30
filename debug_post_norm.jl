using Inferno
using LinearAlgebra

function debug_post_norm()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token through layers 1-6
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    layer = model.layers[7]
    
    # Manually compute SSM output
    x_orig = copy(x)
    x_norm = layer.in_norm(x)
    ssm_out = layer.op(x_norm, pos, model.rope, caches[7])
    
    # Residual
    x_after_ssm = x_orig .+ ssm_out
    
    println("=== Post-norm Analysis ===")
    println("x_after_ssm norm: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    println("x_after_ssm mean: ", round(sum(x_after_ssm) / length(x_after_ssm), digits=3))
    println("x_after_ssm std: ", round(sqrt(sum((x_after_ssm .- sum(x_after_ssm)/length(x_after_ssm)).^2) / length(x_after_ssm)), digits=3))
    
    # Apply post_norm
    x_post_norm = layer.post_norm(x_after_ssm)
    
    println("\nx_post_norm norm: ", round(sqrt(sum(abs2.(x_post_norm))), digits=3))
    println("x_post_norm mean: ", round(sum(x_post_norm) / length(x_post_norm), digits=3))
    
    # Manual RMS norm computation
    ss = sum(abs2, x_after_ssm)
    m = ss / length(x_after_ssm)
    scale = 1.0f0 / sqrt(m + layer.post_norm.eps)
    
    println("\nManual RMS computation:")
    println("  sum(x^2) = ", round(ss, digits=3))
    println("  mean(x^2) = ", round(m, digits=3))
    println("  scale = 1/sqrt(mean + eps) = ", round(scale, digits=3))
    input_norm = round(sqrt(sum(abs2.(x_after_ssm))), digits=3)
    weight_mean = round(sum(layer.post_norm.weight) / length(layer.post_norm.weight), digits=3)
    println("  expected output norm = input_norm * scale * weight_mean")
    println("                        = $input_norm * $(round(scale, digits=3)) * $weight_mean")
    println("                        = ", round(input_norm * scale * weight_mean, digits=3))
    
    # Check if there's an issue with the computation
    println("\nActual vs expected:")
    println("  Actual norm: ", round(sqrt(sum(abs2.(x_post_norm))), digits=3))
    println("  Expected (if uniform): ", round(sqrt(sum(abs2.(x_after_ssm))) * scale * sum(layer.post_norm.weight) / length(layer.post_norm.weight), digits=3))
end

debug_post_norm()
