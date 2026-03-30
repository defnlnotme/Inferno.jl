using Inferno
using LinearAlgebra

function verify_gate_computation()
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
    ssm = layer.op
    x_norm = layer.in_norm(x)
    
    # Get z before any processing
    z = ssm.gate_proj * x_norm
    
    println("=== Gate (z) Analysis ===")
    println("z norm: ", round(sqrt(sum(abs2.(z))), digits=3))
    println("z mean: ", round(sum(z) / length(z), digits=3))
    println("z std: ", round(sqrt(sum((z .- sum(z)/length(z)).^2) / length(z)), digits=3))
    println("z min: ", round(minimum(z), digits=3))
    println("z max: ", round(maximum(z), digits=3))
    
    # Compute silu(z) = z * sigmoid(z)
    sigmoid_z = 1.0f0 ./ (1.0f0 .+ exp.(-z))
    silu_z = z .* sigmoid_z
    
    println("\nsigmoid(z) mean: ", round(sum(sigmoid_z) / length(sigmoid_z), digits=3))
    println("silu(z) norm: ", round(sqrt(sum(abs2.(silu_z))), digits=3))
    println("silu(z) mean: ", round(sum(silu_z) / length(silu_z), digits=3))
    
    # Compare with a normalized vector
    # If y_all after norm has norm ~11, and silu(z) has norm ~37
    # Then output norm should be sqrt(11^2 + 37^2) if they were orthogonal
    # But since they're multiplied element-wise, the norm depends on the correlation
    
    println("\n=== Expected output calculation ===")
    y_norm_after_rms = 11.219  # From trace
    silu_z_norm = 36.853
    
    println("If y_after_rms norm = $y_norm_after_rms")
    println("And silu(z) norm = $silu_z_norm")
    println("Element-wise product norm depends on alignment")
    
    # If perfectly aligned (all same sign): norm = y * silu_z
    # If uncorrelated: norm = sqrt(mean(y^2 * silu_z^2))
    
    # Let's compute what we get
    y_all = zeros(Float32, 2048)
    y_all .= randn(Float32, 2048)
    y_all .= y_all ./ sqrt(sum(abs2.(y_all))) .* y_norm_after_rms
    
    # Simple simulation
    product_aligned = y_norm_after_rms * silu_z_norm / sqrt(2048)
    println("\nIf perfectly aligned, element-wise product contribution: ", round(product_aligned, digits=3))
    
    # Actual computation
    println("\nActual silu(z) values:")
    println("  Positive count: ", sum(silu_z .> 0))
    println("  Negative count: ", sum(silu_z .< 0))
    println("  Near zero count (|x|<0.01): ", sum(abs.(silu_z) .< 0.01))
    
    # Check if silu(z) is mostly positive or negative
    println("\n  silu(z) sum: ", round(sum(silu_z), digits=3))
end

verify_gate_computation()
