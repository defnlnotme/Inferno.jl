using Inferno
using LinearAlgebra

function debug_post_norm()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Check post_norm weight
    layer = model.layers[1]
    post_norm = layer.post_norm
    
    println("=== Post norm weights ===")
    println("Weight sample: ", post_norm.weight[1:10])
    println("Weight mean: ", sum(post_norm.weight) / length(post_norm.weight))
    println("Weight norm: ", sqrt(sum(abs2.(post_norm.weight))))
    
    # Create a test input with norm 1.01
    x = randn(Float32, 1024)
    x = x .* (1.01f0 / sqrt(sum(abs2.(x))))
    
    println("\n=== Test input ===")
    println("Input norm: ", sqrt(sum(abs2.(x))))
    
    # Apply post_norm
    x_normed = post_norm(x)
    
    println("\n=== After post_norm ===")
    println("Output norm: ", sqrt(sum(abs2.(x_normed))))
    
    # Manual computation
    ss = sum(abs2.(x))
    m = ss / length(x)
    scale = 1.0f0 / sqrt(m + post_norm.eps)
    
    println("\n=== Manual computation ===")
    println("Sum of squares: ", ss)
    println("Mean of squares: ", m)
    println("RMS: ", sqrt(m))
    println("Scale: ", scale)
    println("Expected output norm: ", sqrt(sum(abs2.(x .* scale .* post_norm.weight))))
end

debug_post_norm()
