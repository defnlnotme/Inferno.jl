using Inferno
using LinearAlgebra

function compare_ffn()
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
    
    # MLP step by step
    mlp = layer.mlp
    gate = mlp.gate_weight * x_post
    up = mlp.up_weight * x_post
    silu_gate = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
    hidden = silu_gate .* up
    output = mlp.down_weight * hidden
    
    println("=== Layer 6 FFN Comparison ===")
    println("\nllama.cpp | Our impl | Match?")
    println("---------|----------|-------")
    println("ffn_gate: 35.623 | gate: $(round(sqrt(sum(abs2.(gate))), digits=3)) | $(abs(35.623 - sqrt(sum(abs2.(gate)))) < 0.1 ? "YES" : "NO")")
    println("ffn_up: 14.513 | up: $(round(sqrt(sum(abs2.(up))), digits=3)) | $(abs(14.513 - sqrt(sum(abs2.(up)))) < 0.1 ? "YES" : "NO")")
    println("ffn_swiglu: 2.696 | hidden: $(round(sqrt(sum(abs2.(hidden))), digits=3)) | $(abs(2.696 - sqrt(sum(abs2.(hidden)))) < 0.1 ? "YES" : "NO")")
    println("ffn_out: 0.787 | output: $(round(sqrt(sum(abs2.(output))), digits=3)) | $(abs(0.787 - sqrt(sum(abs2.(output)))) < 0.1 ? "YES" : "NO")")
    
    # Final output
    x_final = x_after_ssm .+ output
    println("l_out: 2.765 | x_final: $(round(sqrt(sum(abs2.(x_final))), digits=3)) | $(abs(2.765 - sqrt(sum(abs2.(x_final)))) < 0.1 ? "YES" : "NO")")
    
    # Check the ratios
    println("\n=== Ratio Analysis ===")
    println("llama.cpp ffn_gate / ffn_up = ", round(35.623 / 14.513, digits=3))
    println("Our gate / up = ", round(sqrt(sum(abs2.(gate))) / sqrt(sum(abs2.(up))), digits=3))
    
    println("\nllama.cpp ffn_swiglu / ffn_gate = ", round(2.696 / 35.623, digits=4))
    println("Our hidden / gate = ", round(sqrt(sum(abs2.(hidden))) / sqrt(sum(abs2.(gate))), digits=4))
end

compare_ffn()
