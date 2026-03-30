using Inferno
using LinearAlgebra

function check_state_persistence()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    
    # Check initial state for layer 7
    ssm7 = model.layers[7].op
    println("Layer 7 SSM state BEFORE processing:")
    println("  h norm: ", sqrt(sum(abs2.(ssm7.h))))
    println("  conv_state norm: ", sqrt(sum(abs2.(ssm7.conv_state))))
    
    # Process first 7 layers
    x = model.embed[:, tok]
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    # Check state after processing layers 1-6
    println("\nLayer 7 SSM state AFTER processing layers 1-6:")
    println("  h norm: ", sqrt(sum(abs2.(ssm7.h))))
    println("  conv_state norm: ", sqrt(sum(abs2.(ssm7.conv_state))))
    
    # Process layer 7
    x = model.layers[7](x, pos, model.rope, caches[7])
    
    # Check state after processing layer 7
    println("\nLayer 7 SSM state AFTER processing layer 7:")
    println("  h norm: ", sqrt(sum(abs2.(ssm7.h))))
    println("  conv_state norm: ", sqrt(sum(abs2.(ssm7.conv_state))))
    
    # Process layer 8
    x = model.layers[8](x, pos, model.rope, caches[8])
    
    # Check state after processing layer 8
    println("\nLayer 7 SSM state AFTER processing layer 8:")
    println("  h norm: ", sqrt(sum(abs2.(ssm7.h))))
    println("  conv_state norm: ", sqrt(sum(abs2.(ssm7.conv_state))))
end

check_state_persistence()
