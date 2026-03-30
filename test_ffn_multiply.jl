using Inferno
using LinearAlgebra

function test_ffn_multiply()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    mlp6 = model.layers[6].mlp
    
    println("=== FFN Weight Shapes ===")
    println("gate_weight: ", size(mlp6.gate_weight))
    println("up_weight: ", size(mlp6.up_weight))
    println("down_weight: ", size(mlp6.down_weight))
    
    # Create a test input
    x_post = zeros(Float32, 1024)  # hidden_size
    x_post[1:10] .= 1.0
    
    println("\n=== Testing Multiplications ===")
    println("Input shape: ", size(x_post))
    
    # Test gate multiplication
    println("\nTesting gate_weight * x_post:")
    println("  gate_weight: ", size(mlp6.gate_weight))
    println("  x_post: ", size(x_post))
    gate = mlp6.gate_weight * x_post
    println("  gate result shape: ", size(gate))
    
    # Test up multiplication
    println("\nTesting up_weight * x_post:")
    up = mlp6.up_weight * x_post
    println("  up result shape: ", size(up))
    
    # Test hidden (gate_silu .* up)
    gate_silu = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
    hidden = gate_silu .* up
    println("\nHidden shape: ", size(hidden))
    
    # Test down multiplication
    println("\nTesting down_weight * hidden:")
    println("  down_weight: ", size(mlp6.down_weight))
    println("  hidden: ", size(hidden))
    output = mlp6.down_weight * hidden
    println("  output result shape: ", size(output))
end

test_ffn_multiply()
