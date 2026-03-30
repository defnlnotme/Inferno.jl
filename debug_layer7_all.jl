using Inferno
using LinearAlgebra

function trace_layer7_all_tokens()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    prompt_tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    println("=== Processing tokens through layer 7 ===")
    
    for (pos, tok) in enumerate(prompt_tokens)
        # Reset caches for fair comparison
        Inferno.ModelCPU.reset_states_cpu!(model)
        for c in caches
            fill!(c.k, 0.0f0)
            fill!(c.v, 0.0f0)
        end
        
        x = model.embed[:, tok]
        
        # Process through layers 1-6
        for i in 1:6
            x = model.layers[i](x, pos-1, model.rope, caches[i])
        end
        
        # Get layer 7 details
        layer = model.layers[7]
        x_norm = layer.in_norm(x)
        ssm_out = layer.op(x_norm, pos-1, model.rope, caches[7])
        
        println("\nToken $pos ($tok):")
        println("  Input norm: ", round(sqrt(sum(abs2.(x))), digits=3))
        println("  SSM output norm: ", round(sqrt(sum(abs2.(ssm_out))), digits=3))
    end
    
    # Now process full prompt and check layer 7 output for last position
    println("\n=== Full prompt processing ===")
    Inferno.ModelCPU.reset_states_cpu!(model)
    for c in caches
        fill!(c.k, 0.0f0)
        fill!(c.v, 0.0f0)
    end
    
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(prompt_tokens)
        x = model.embed[:, tok]
        for i in 1:7
            x = model.layers[i](x, pos-1, model.rope, caches[i])
        end
        println("After layer 7 for token $pos: norm = ", round(sqrt(sum(abs2.(x))), digits=3))
    end
end

trace_layer7_all_tokens()
