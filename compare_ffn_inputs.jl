using Inferno
using LinearAlgebra

function compare_ffn_inputs()
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
    
    println("=== Post-Norm Comparison ===")
    println("llama.cpp attn_post_norm-5: 37.505")
    println("Our x_post: ", round(sqrt(sum(abs2.(x_post))), digits=3))
    println("Match: ", abs(37.505 - sqrt(sum(abs2.(x_post)))) < 0.1 ? "YES" : "NO")
    
    # Now compute FFN projections
    mlp = layer.mlp
    gate = mlp.gate_weight * x_post
    up = mlp.up_weight * x_post
    
    println("\n=== FFN Projections ===")
    println("llama.cpp ffn_gate-5: 35.623")
    println("Our gate: ", round(sqrt(sum(abs2.(gate))), digits=3))
    println("Match: ", abs(35.623 - sqrt(sum(abs2.(gate)))) < 0.1 ? "YES" : "NO")
    
    println("\nllama.cpp ffn_up-5: 14.513")
    println("Our up: ", round(sqrt(sum(abs2.(up))), digits=3))
    println("Match: ", abs(14.513 - sqrt(sum(abs2.(up)))) < 0.1 ? "YES" : "NO")
    
    # Check weight multiplication direction
    # In llama.cpp: ffn_gate = gate_weight * input (matrix-vector multiply)
    # In our impl: gate = gate_weight * x_post (matrix-vector multiply)
    # 
    # Let me check if the multiplication is correct
    println("\n=== Weight Shapes ===")
    println("gate_weight: ", size(mlp.gate_weight), " (should be (intermediate, hidden))")
    println("up_weight: ", size(mlp.up_weight), " (should be (intermediate, hidden))")
    
    # Check if weight is transposed
    println("\n=== Matrix-Vector Multiply Check ===")
    println("gate_weight * x_post: shape ", size(gate), " (should be (intermediate,))")
    
    # Try transposed multiply
    gate_T = x_post' * mlp.gate_weight'
    println("x_post' * gate_weight': shape ", size(gate_T), " (should be (1, intermediate))")
    
    # Check which one matches llama.cpp
    println("\n=== Norm Check ===")
    println("gate = gate_weight * x_post: norm = ", round(sqrt(sum(abs2.(gate))), digits=3))
    println("gate_T = x_post' * gate_weight': norm = ", round(sqrt(sum(abs2.(gate_T))), digits=3))
    println("llama.cpp ffn_gate-5: norm = 35.623")
    
    # The correct one should be closer to 35.623
    if abs(35.623 - sqrt(sum(abs2.(gate)))) < abs(35.623 - sqrt(sum(abs2.(gate_T))))
        println("\n=> gate_weight * x_post is correct")
    else
        println("\n=> x_post' * gate_weight' is correct (transpose needed)")
    end
end

compare_ffn_inputs()
