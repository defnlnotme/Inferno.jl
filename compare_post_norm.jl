using Inferno
using LinearAlgebra

function compare_post_norm()
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
    
    println("=== Post-Norm Investigation for Layer 6 ===")
    println("Input to layer 6: norm = ", round(sqrt(sum(abs2.(x))), digits=3))
    
    layer = model.layers[6]
    
    # Apply in_norm
    x_norm = layer.in_norm(x)
    println("After in_norm: norm = ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    
    # SSM output
    ssm_output = layer.op(x_norm, pos, model.rope, caches[6])
    println("SSM output: norm = ", round(sqrt(sum(abs2.(ssm_output))), digits=3))
    
    # Residual
    x_after_ssm = x .+ ssm_output
    println("After residual: norm = ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    
    # Post norm
    x_post = layer.post_norm(x_after_ssm)
    println("After post_norm: norm = ", round(sqrt(sum(abs2.(x_post))), digits=3))
    
    # Compare with llama.cpp
    println("\n=== Comparison with llama.cpp ===")
    println("llama.cpp:")
    println("  attn_norm-5: 47.57 (input to layer after RMS norm)")
    println("  attn_residual-5: 2.80 (input + SSM output)")
    println("Our implementation:")
    println("  x_norm (attn_norm): ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    println("  x_after_ssm (attn_residual): ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    
    # The key difference: llama.cpp's attn_norm is ~47-48, ours should be similar
    
    # Check RMS norm computation
    println("\n=== Post-Norm Analysis ===")
    println("Input to post_norm:")
    println("  x_after_ssm norm: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    println("  x_after_ssm mean: ", round(sum(x_after_ssm) / length(x_after_ssm), digits=6))
    
    ss = sum(abs2, x_after_ssm)
    m = ss / length(x_after_ssm)
    scale = 1.0f0 / sqrt(m + layer.post_norm.eps)
    
    println("\nRMS norm computation:")
    println("  sum(x^2) = ", round(ss, digits=3))
    println("  mean(x^2) = ", round(m, digits=6))
    println("  scale = 1/sqrt(mean + eps) = ", round(scale, digits=3))
    
    println("\nPost-norm weight:")
    println("  mean: ", round(sum(layer.post_norm.weight) / length(layer.post_norm.weight), digits=3))
    
    println("\nExpected output norm:")
    expected_norm = round(sqrt(sum(abs2.(x_after_ssm))) * scale * sum(layer.post_norm.weight) / length(layer.post_norm.weight), digits=3)
    println("  input_norm * scale * weight_mean = ", expected_norm)
    println("Actual output norm: ", round(sqrt(sum(abs2.(x_post))), digits=3))
end

compare_post_norm()
