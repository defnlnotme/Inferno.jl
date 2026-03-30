using Inferno
using LinearAlgebra

function compare_ffn_outputs()
    # Load model
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Use layer 6 (1-indexed: model.layers[7])
    mlp6 = model.layers[7].mlp
    
    println("=== FFN Weight Norms (Layer 6) ===")
    println("gate_weight norm: ", round(norm(mlp6.gate_weight), digits=3))
    println("up_weight norm: ", round(norm(mlp6.up_weight), digits=3))
    println("down_weight norm: ", round(norm(mlp6.down_weight), digits=3))
    
    # Create test input matching llama.cpp
    x_post = zeros(Float32, 1024)
    x_post[1:100] .= 0.1f0
    x_post[101:200] .= -0.05f0
    
    println("\nInput norm: ", round(norm(x_post), digits=3))
    
    # Run FFN
    gate = mlp6.gate_weight * x_post
    gate_silu = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))  # SiLU activation
    up = mlp6.up_weight * x_post
    hidden = gate_silu .* up
    output = mlp6.down_weight * hidden
    
    println("\n=== Our Results ===")
    println("gate norm: ", round(norm(gate), digits=3))
    println("up norm: ", round(norm(up), digits=3))
    println("hidden norm: ", round(norm(hidden), digits=3))
    println("output norm: ", round(norm(output), digits=3))
    
    # Compare with llama.cpp (from earlier run)
    println("\n=== Comparison with llama.cpp ===")
    println("llama.cpp ffn_gate norm: 35.623")
    println("Our gate norm: ", round(norm(gate), digits=3))
    println("")
    println("llama.cpp ffn_up norm: 14.513")
    println("Our up norm: ", round(norm(up), digits=3))
    
    ratio_ours = norm(gate) / norm(up)
    ratio_llama = 35.623 / 14.513
    println("\n=== Ratio Analysis ===")
    println("Our gate/up ratio: ", round(ratio_ours, digits=3))
    println("llama.cpp gate/up ratio: ", round(ratio_llama, digits=3))
end

compare_ffn_outputs()
