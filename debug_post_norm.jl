using Inferno
using LinearAlgebra

function debug_post_norm()
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
    
    println("=== RMS Norm Debug ===")
    println("\nInput to post_norm (x_after_ssm):")
    println("  norm: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    println("  size: ", length(x_after_ssm))
    
    # Manual RMS norm computation
    ss = sum(abs2, x_after_ssm)
    mean_sq = ss / length(x_after_ssm)
    rms = sqrt(mean_sq)
    scale = 1.0f0 / (rms + 1e-6)  # RMS norm with epsilon
    
    println("\nRMS norm computation (manual):")
    println("  sum(x^2) = ", round(ss, digits=3))
    println("  mean(x^2) = ", round(mean_sq, digits=6))
    println("  sqrt(mean) = ", round(rms, digits=6))
    println("  scale = 1/(sqrt(mean) + eps) = ", round(scale, digits=3))
    
    # Apply scale
    x_scaled = x_after_ssm .* scale
    println("\nAfter scaling:")
    println("  x_scaled norm: ", round(sqrt(sum(abs2.(x_scaled))), digits=3))
    
    # Apply weight
    println("\nPost-norm weight:")
    println("  mean: ", round(sum(layer.post_norm.weight) / length(layer.post_norm.weight), digits=3))
    println("  norm: ", round(sqrt(sum(abs2.(layer.post_norm.weight))), digits=3))
    
    x_post_manual = x_scaled .* layer.post_norm.weight
    println("\nAfter weight multiplication (manual):")
    println("  x_post norm: ", round(sqrt(sum(abs2.(x_post_manual))), digits=3))
    
    # Compare with actual
    x_post_actual = layer.post_norm(x_after_ssm)
    println("\nActual post_norm output:")
    println("  x_post norm: ", round(sqrt(sum(abs2.(x_post_actual))), digits=3))
    
    # Check if they match
    println("\nDifference between manual and actual: ", round(sqrt(sum(abs2.(x_post_manual - x_post_actual))), digits=6))
    
    # Check llama.cpp expected values
    println("\n=== llama.cpp Comparison ===")
    println("llama.cpp:")
    println("  attn_residual-5: 2.80")
    println("  ffn_gate-5: 35.623 (gate projection)")
    println("  ffn_up-5: 14.513 (up projection)")
    println("")
    println("Expected behavior:")
    println("  norm(ffn_gate) / norm(gate_weight) ≈ norm(ffn_up) / norm(up_weight)")
    println("  35.623 / 27.936 = ", round(35.623 / 27.936, digits=3))
    println("  14.513 / 15.874 = ", round(14.513 / 15.874, digits=3))
    println("")
    println("These should be similar if the input to FFN is the same!")
    println("But 1.275 ≠ 0.914, so there's something wrong.")
end

debug_post_norm()
