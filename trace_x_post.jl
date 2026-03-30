using Inferno
using LinearAlgebra

function trace_x_post()
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
    
    println("=== Input to Layer 6 ===")
    println("x norm: ", round(sqrt(sum(abs2.(x))), digits=3))
    
    layer = model.layers[6]
    
    # Apply in_norm
    x_norm = layer.in_norm(x)
    println("\n=== After in_norm ===")
    println("x_norm norm: ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    
    # SSM output
    ssm_output = layer.op(x_norm, pos, model.rope, caches[6])
    println("\n=== SSM output ===")
    println("ssm_output norm: ", round(sqrt(sum(abs2.(ssm_output))), digits=3))
    
    # Residual
    x_after_ssm = x .+ ssm_output
    println("\n=== After residual ===")
    println("x_after_ssm norm: ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    
    # Post norm
    x_post = layer.post_norm(x_after_ssm)
    println("\n=== After post_norm (input to FFN) ===")
    println("x_post norm: ", round(sqrt(sum(abs2.(x_post))), digits=3))
    
    # Compare with llama.cpp
    println("\n=== Comparison with llama.cpp ===")
    println("llama.cpp:")
    println("  attn_norm-5: 47.57 (input to layer after in_norm)")
    println("  attn_residual-5: 2.80 (after SSM residual)")
    println("Our impl:")
    println("  x_norm: $(round(sqrt(sum(abs2.(x_norm))), digits=3))")
    println("  x_after_ssm: $(round(sqrt(sum(abs2.(x_after_ssm))), digits=3))")
    
    # The key difference: x_post is the input to FFN
    # Let me check what llama.cpp uses as input to FFN
    # From the dump: ffn_gate-5 norm = 35.623, ffn_up-5 norm = 14.513
    # These are projections of attn_post_norm
    # Let me compute what the input should be
    
    # gate = gate_weight * input
    # norm(gate) = norm(gate_weight) * norm(input) * cos(angle)
    # If we assume the angle is roughly the same:
    # norm(input) ≈ norm(gate) / norm(gate_weight)
    
    gate_weight_norm = sqrt(sum(abs2.(layer.mlp.gate_weight)))
    up_weight_norm = sqrt(sum(abs2.(layer.mlp.up_weight)))
    
    println("\n=== Weight norms ===")
    println("gate_weight norm: ", round(gate_weight_norm, digits=3))
    println("up_weight norm: ", round(up_weight_norm, digits=3))
    
    # Estimate llama.cpp's input norm
    llama_gate_norm = 35.623
    llama_up_norm = 14.513
    
    println("\n=== Estimated llama.cpp input ===")
    println("From gate: norm(input) ≈ norm(gate) / norm(gate_weight) = ", round(llama_gate_norm / gate_weight_norm, digits=3))
    println("From up: norm(input) ≈ norm(up) / norm(up_weight) = ", round(llama_up_norm / up_weight_norm, digits=3))
    println("Our x_post: ", round(sqrt(sum(abs2.(x_post))), digits=3))
    
    # The ratio difference
    println("\n=== Ratio Analysis ===")
    println("llama.cpp gate/up ratio: ", round(llama_gate_norm / llama_up_norm, digits=3))
    println("Our gate/up ratio: ", round(gate_weight_norm / up_weight_norm, digits=3))
end

trace_x_post()
