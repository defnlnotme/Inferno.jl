using Inferno
using LinearAlgebra

function compare_weight_norms()
    # Load model
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Use layer 6 (1-indexed: model.layers[7])
    mlp6 = model.layers[7].mlp
    
    println("=== FFN Weight Norms (Layer 6) ===")
    gate_norm = norm(mlp6.gate_weight)
    up_norm = norm(mlp6.up_weight)
    
    println("gate_weight norm: ", round(gate_norm, digits=3))
    println("up_weight norm: ", round(up_norm, digits=3))
    println("gate/up ratio: ", round(gate_norm / up_norm, digits=3))
    
    # These weight norms should match llama.cpp
    # Let me also check if the weights are transposed correctly
    println("\n=== Weight shapes ===")
    println("gate_weight: ", size(mlp6.gate_weight), " (expected: 3584 x 1024)")
    println("up_weight: ", size(mlp6.up_weight), " (expected: 3584 x 1024)")
    println("down_weight: ", size(mlp6.down_weight), " (expected: 1024 x 3584)")
end

compare_weight_norms()
