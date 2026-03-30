using Inferno
using LinearAlgebra

function trace_attention_layer4()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token "The" through layers 1-3
    x = model.embed[:, 761]
    pos = 0
    
    for i in 1:3
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    println("=== Input to attention layer 4 ===")
    println("Input norm: ", sqrt(sum(abs2.(x))))
    
    # Process through attention layer 4 manually
    layer = model.layers[4]
    attn = layer.op
    
    # Step 1: Input norm
    x_norm = layer.in_norm(x)
    println("\nAfter in_norm: ", sqrt(sum(abs2.(x_norm))))
    
    # Step 2: Q, K, V projections
    qkv = attn.wq * x_norm
    k = attn.wk * x_norm
    v = attn.wv * x_norm
    
    println("\n=== Projections ===")
    println("qkv shape: ", size(qkv))
    println("qkv norm: ", sqrt(sum(abs2.(qkv))))
    println("k norm: ", sqrt(sum(abs2.(k))))
    println("v norm: ", sqrt(sum(abs2.(v))))
    
    # Split qkv into query and gate
    q_size = attn.n_heads * attn.head_dim
    query_states = qkv[1:q_size]
    gate = qkv[q_size+1:end]
    
    println("\n=== Query and Gate ===")
    println("query norm: ", sqrt(sum(abs2.(query_states))))
    println("gate norm: ", sqrt(sum(abs2.(gate))))
    
    # Reshape
    query_states = reshape(query_states, attn.head_dim, attn.n_heads)
    k = reshape(k, attn.head_dim, attn.n_kv)
    v = reshape(v, attn.head_dim, attn.n_kv)
    
    println("\n=== After reshape ===")
    println("query shape: ", size(query_states))
    println("k shape: ", size(k))
    println("v shape: ", size(v))
    
    # Apply Q/K normalization
    for h in 1:attn.n_heads
        q_h = view(query_states, :, h)
        Inferno.ModelCPU.rmsnorm_cpu!(q_h, q_h, attn.q_norm)
    end
    for h in 1:attn.n_kv
        k_h = view(k, :, h)
        Inferno.ModelCPU.rmsnorm_cpu!(k_h, k_h, attn.k_norm)
    end
    
    println("\n=== After Q/K norm ===")
    println("query norm: ", sqrt(sum(abs2.(query_states))))
    println("k norm: ", sqrt(sum(abs2.(k))))
    
    # Apply sigmoid gating
    gate_sigmoid = 1.0f0 ./ (1.0f0 .+ exp.(-gate))
    gate_reshaped = reshape(gate_sigmoid, attn.head_dim, attn.n_heads)
    query_states .*= gate_reshaped
    
    println("\n=== After sigmoid gate ===")
    println("query norm: ", sqrt(sum(abs2.(query_states))))
    println("gate_sigmoid mean: ", sum(gate_sigmoid) / length(gate_sigmoid))
end

trace_attention_layer4()
