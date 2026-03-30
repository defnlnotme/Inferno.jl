using Inferno
using LinearAlgebra

function trace_all_layers()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    
    println("Layer-by-layer norm trace:")
    println("Embedding: norm = ", round(sqrt(sum(abs2.(x))), digits=3))
    
    for i in 1:model.config.num_hidden_layers
        x_before = copy(x)
        x = model.layers[i](x, pos, model.rope, caches[i])
        
        layer_type = model.layers[i].is_ssm ? "SSM" : "Attn"
        println("Layer $i ($layer_type): norm = ", round(sqrt(sum(abs2.(x))), digits=3), 
                " (delta = ", round(sqrt(sum(abs2.(x))) - sqrt(sum(abs2.(x_before))), digits=3), ")")
    end
    
    println("\nFinal norm: ", round(sqrt(sum(abs2.(x))), digits=3))
    
    # Compare with llama.cpp expected behavior
    # For a well-trained model, the norm should stay relatively stable
end

trace_all_layers()
