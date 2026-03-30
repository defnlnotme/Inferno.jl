using Inferno
using LinearAlgebra

function debug_mlp()
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
    
    # Manually compute up to post_norm
    x_orig = copy(x)
    x_norm = layer.in_norm(x)
    ssm_out = layer.op(x_norm, pos, model.rope, caches[7])
    x_after_ssm = x_orig .+ ssm_out
    x_post_norm = layer.post_norm(x_after_ssm)
    
    println("=== MLP Analysis ===")
    println("Input to MLP (x_post_norm) norm: ", round(sqrt(sum(abs2.(x_post_norm))), digits=3))
    
    # Manual MLP computation
    mlp = layer.mlp
    
    # Gate projection with SiLU
    gate = mlp.gate_weight * x_post_norm
    gate_silu = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
    
    println("\nGate projection:")
    println("  gate norm: ", round(sqrt(sum(abs2.(gate))), digits=3))
    println("  gate_silu norm: ", round(sqrt(sum(abs2.(gate_silu))), digits=3))
    
    # Up projection
    up = mlp.up_weight * x_post_norm
    println("\nUp projection:")
    println("  up norm: ", round(sqrt(sum(abs2.(up))), digits=3))
    
    # Element-wise product
    hidden = gate_silu .* up
    println("\nHidden (gate_silu .* up):")
    println("  hidden norm: ", round(sqrt(sum(abs2.(hidden))), digits=3))
    
    # Down projection
    output = mlp.down_weight * hidden
    println("\nDown projection (MLP output):")
    println("  output norm: ", round(sqrt(sum(abs2.(output))), digits=3))
    
    println("\n=== MLP Weight Norms ===")
    println("gate_weight norm: ", round(sqrt(sum(abs2.(mlp.gate_weight))), digits=3))
    println("up_weight norm: ", round(sqrt(sum(abs2.(mlp.up_weight))), digits=3))
    println("down_weight norm: ", round(sqrt(sum(abs2.(mlp.down_weight))), digits=3))
    
    println("\n=== Residual After MLP ===")
    x_final = x_after_ssm .+ output
    println("x_final norm: ", round(sqrt(sum(abs2.(x_final))), digits=3))
end

debug_mlp()
