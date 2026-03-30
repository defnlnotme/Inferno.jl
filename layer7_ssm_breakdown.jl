using Inferno
using LinearAlgebra

function layer7_ssm_breakdown()
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
    
    println("=== First token SSM breakdown for layer 7 ===")
    println("Input norm: ", round(sqrt(sum(abs2.(x))), digits=3))
    
    layer = model.layers[7]
    ssm = layer.op
    
    # Get SSM components
    x_norm = layer.in_norm(x)
    
    # In projections
    in_proj = ssm.in_proj * x_norm
    gate_proj = ssm.gate_proj * x_norm
    
    println("\nIn projection norm: ", round(sqrt(sum(abs2.(in_proj))), digits=3))
    println("Gate projection norm: ", round(sqrt(sum(abs2.(gate_proj))), digits=3))
    
    # Check gate values
    gate_sigmoid = 1.0f0 ./ (1.0f0 .+ exp.(-gate_proj))
    println("Gate sigmoid mean: ", round(sum(gate_sigmoid) / length(gate_sigmoid), digits=3))
    println("Gate sigmoid min: ", round(minimum(gate_sigmoid), digits=3))
    println("Gate sigmoid max: ", round(maximum(gate_sigmoid), digits=3))
end

layer7_ssm_breakdown()
