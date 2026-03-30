using Inferno
using LinearAlgebra
using Random

function test_random_init()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Test 1: Zero initialization (current behavior)
    println("=== Test 1: Zero Initialization ===")
    caches_zero = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    x_zero = model.embed[:, tok]
    
    for i in 1:model.config.num_hidden_layers
        x_zero = model.layers[i](x_zero, pos, model.rope, caches_zero[i])
    end
    
    println("Final norm (zero init): ", round(sqrt(sum(abs2.(x_zero))), digits=3))
    
    # Test 2: Random initialization
    println("\n=== Test 2: Random Initialization ===")
    
    # Reinitialize model with random states
    Random.seed!(42)
    
    caches_random = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    
    # Initialize delta net states with small random values
    for i in 1:model.config.num_hidden_layers
        layer = model.layers[i]
        if layer.is_ssm
            ssm = layer.op
            # Initialize h with small random values
            ssm.h .= randn(Float32, size(ssm.h)) .* 0.01f0
            # Initialize conv_state with small random values
            ssm.conv_state .= randn(Float32, size(ssm.conv_state)) .* 0.01f0
        end
    end
    
    x_random = model.embed[:, tok]
    
    for i in 1:model.config.num_hidden_layers
        x_random = model.layers[i](x_random, pos, model.rope, caches_random[i])
    end
    
    println("Final norm (random init): ", round(sqrt(sum(abs2.(x_random))), digits=3))
    
    # Test 3: Small constant initialization
    println("\n=== Test 3: Small Constant Initialization ===")
    
    caches_const = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    
    for i in 1:model.config.num_hidden_layers
        layer = model.layers[i]
        if layer.is_ssm
            ssm = layer.op
            ssm.h .= 0.01f0
            ssm.conv_state .= 0.01f0
        end
    end
    
    x_const = model.embed[:, tok]
    
    for i in 1:model.config.num_hidden_layers
        x_const = model.layers[i](x_const, pos, model.rope, caches_const[i])
    end
    
    println("Final norm (const init): ", round(sqrt(sum(abs2.(x_const))), digits=3))
    
    # Compare layer-by-layer
    println("\n=== Layer-by-Layer Comparison ===")
    
    # Reset and run with zero init
    Inferno.ModelCPU.reset_states_cpu!(model)
    x_z = model.embed[:, tok]
    norms_zero = Float32[]
    for i in 1:model.config.num_hidden_layers
        x_z = model.layers[i](x_z, pos, model.rope, caches_zero[i])
        push!(norms_zero, sqrt(sum(abs2.(x_z))))
    end
    
    # Reset and run with random init
    Random.seed!(42)
    for i in 1:model.config.num_hidden_layers
        layer = model.layers[i]
        if layer.is_ssm
            ssm = layer.op
            ssm.h .= randn(Float32, size(ssm.h)) .* 0.01f0
            ssm.conv_state .= randn(Float32, size(ssm.conv_state)) .* 0.01f0
        end
    end
    
    x_r = model.embed[:, tok]
    norms_random = Float32[]
    for i in 1:model.config.num_hidden_layers
        x_r = model.layers[i](x_r, pos, model.rope, caches_random[i])
        push!(norms_random, sqrt(sum(abs2.(x_r))))
    end
    
    println("Layer | Zero Init | Random Init")
    println("------|-----------|------------")
    for i in 1:model.config.num_hidden_layers
        layer_type = model.layers[i].is_ssm ? "SSM" : "Attn"
        println("  $i ($layer_type) | $(round(norms_zero[i], digits=3)) | $(round(norms_random[i], digits=3))")
    end
end

test_random_init()
