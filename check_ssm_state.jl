using Inferno
using LinearAlgebra

function check_ssm_state_accumulation()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    prompt_tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    println("=== SSM State Accumulation ===")
    println("Token | SSM State Norm (Layer 1)")
    println("------|-------------------------")
    
    for (pos, tok) in enumerate(prompt_tokens)
        # Check SSM state before processing
        ssm = model.layers[1].op
        state_norm_before = sqrt(sum(abs2.(ssm.h)))
        
        # Process token
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
        
        # Check SSM state after processing
        state_norm_after = sqrt(sum(abs2.(ssm.h)))
        
        tok_str = ""  # Skip token decoding
        println("$pos | Before: $(round(state_norm_before, digits=3)) | After: $(round(state_norm_after, digits=3))")
    end
    
    # Also check how the SSM state changes per layer
    println("\n=== Final SSM State per Layer ===")
    for i in 1:3  # First 3 SSM layers
        ssm = model.layers[i].op
        state_norm = sqrt(sum(abs2.(ssm.h)))
        println("Layer $i: SSM state norm = $(round(state_norm, digits=3))")
    end
end

check_ssm_state_accumulation()
