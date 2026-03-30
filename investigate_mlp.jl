using Inferno
using LinearAlgebra

function investigate_mlp()
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
    
    println("=== MLP Investigation for Layer 6 ===")
    
    layer = model.layers[6]
    
    # Process through SSM
    x_before = copy(x)
    x_norm = layer.in_norm(x)
    ssm_output = layer.op(x_norm, pos, model.rope, caches[6])
    x_after_ssm = x_before .+ ssm_output
    
    println("After SSM + residual: norm = ", round(sqrt(sum(abs2.(x_after_ssm))), digits=3))
    
    # Apply post_norm
    x_post = layer.post_norm(x_after_ssm)
    println("After post_norm: norm = ", round(sqrt(sum(abs2.(x_post))), digits=3))
    
    # Now trace MLP step by step
    mlp = layer.mlp
    
    println("\n=== MLP Step-by-Step ===")
    println("Input to MLP (x_post) norm: ", round(sqrt(sum(abs2.(x_post))), digits=3))
    
    # Gate projection
    gate = mlp.gate_weight * x_post
    println("\n1. Gate projection:")
    println("   gate_weight shape: ", size(mlp.gate_weight))
    println("   gate_weight norm: ", round(sqrt(sum(abs2.(mlp.gate_weight))), digits=3))
    println("   gate output norm: ", round(sqrt(sum(abs2.(gate))), digits=3))
    println("   gate sample: ", round.(gate[1:5], digits=4))
    
    # SiLU on gate
    gate_silu = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
    println("\n2. SiLU(gate):")
    println("   gate_silu norm: ", round(sqrt(sum(abs2.(gate_silu))), digits=3))
    println("   gate_silu sample: ", round.(gate_silu[1:5], digits=4))
    
    # Up projection
    up = mlp.up_weight * x_post
    println("\n3. Up projection:")
    println("   up_weight shape: ", size(mlp.up_weight))
    println("   up_weight norm: ", round(sqrt(sum(abs2.(mlp.up_weight))), digits=3))
    println("   up output norm: ", round(sqrt(sum(abs2.(up))), digits=3))
    println("   up sample: ", round.(up[1:5], digits=4))
    
    # Element-wise product
    hidden = gate_silu .* up
    println("\n4. Hidden (gate_silu .* up):")
    println("   hidden norm: ", round(sqrt(sum(abs2.(hidden))), digits=3))
    println("   hidden sample: ", round.(hidden[1:5], digits=4))
    
    # Down projection
    output = mlp.down_weight * hidden
    println("\n5. Down projection:")
    println("   down_weight shape: ", size(mlp.down_weight))
    println("   down_weight norm: ", round(sqrt(sum(abs2.(mlp.down_weight))), digits=3))
    println("   output norm: ", round(sqrt(sum(abs2.(output))), digits=3))
    println("   output sample: ", round.(output[1:5], digits=4))
    
    println("\n=== Comparison with llama.cpp ===")
    println("llama.cpp ffn_out-5: 0.79")
    println("Our MLP output: ", round(sqrt(sum(abs2.(output))), digits=3))
    
    # Check if the weights are loaded correctly
    println("\n=== Weight Analysis ===")
    println("gate_weight: shape=$(size(mlp.gate_weight)), norm=$(round(sqrt(sum(abs2.(mlp.gate_weight))), digits=3))")
    println("up_weight: shape=$(size(mlp.up_weight)), norm=$(round(sqrt(sum(abs2.(mlp.up_weight))), digits=3))")
    println("down_weight: shape=$(size(mlp.down_weight)), norm=$(round(sqrt(sum(abs2.(mlp.down_weight))), digits=3))")
    
    # Expected shapes: (intermediate_size, hidden_size) for gate/up, (hidden_size, intermediate_size) for down
    println("\nExpected shapes:")
    println("  gate_weight: (intermediate_size, hidden_size) = (3584, 1024)")
    println("  up_weight: (intermediate_size, hidden_size) = (3584, 1024)")
    println("  down_weight: (hidden_size, intermediate_size) = (1024, 3584)")
end

investigate_mlp()
