using Inferno
using LinearAlgebra

function debug_silu_formula()
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
    
    # Get z
    z = ssm.gate_proj * x_norm
    
    println("=== SiLU Formula Check ===")
    println("\nz sample values:")
    for i in [1, 100, 500, 1000, 1500, 2000]
        println("  z[$i] = ", round(z[i], digits=4))
    end
    
    # Manual SiLU computation
    # silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
    
    println("\nManual SiLU computation:")
    for i in [1, 100, 500, 1000, 1500, 2000]
        zi = z[i]
        sigmoid_zi = 1.0f0 / (1.0f0 + exp(-zi))
        silu_zi = zi * sigmoid_zi
        println("  silu(z[$i]) = z * sigmoid(z) = $zi * $sigmoid_zi = $silu_zi")
    end
    
    # Verify the gate_proj weights
    println("\n=== Gate Projection Weights ===")
    println("gate_proj shape: ", size(ssm.gate_proj))
    println("gate_proj norm: ", round(sqrt(sum(abs2.(ssm.gate_proj))), digits=3))
    
    # Check if the weights are reasonable
    println("gate_proj mean abs: ", round(sum(abs.(ssm.gate_proj)) / length(ssm.gate_proj), digits=6))
    
    # Check x_norm
    println("\nx_norm norm: ", round(sqrt(sum(abs2.(x_norm))), digits=3))
    println("x_norm sample: ", x_norm[1:5])
end

debug_silu_formula()
