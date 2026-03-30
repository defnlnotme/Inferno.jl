using Inferno
using LinearAlgebra

function trace_attention_full()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token "The" through layers 1-3
    x = model.embed[:, 761]
    pos = 0
    
    for i in 1:3
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    # Get attention layer 4
    layer = model.layers[4]
    attn = layer.op
    cache = caches[4]
    
    # Process through attention manually
    x_norm = layer.in_norm(x)
    
    # Q, K, V projections
    qkv = attn.wq * x_norm
    k = attn.wk * x_norm
    v = attn.wv * x_norm
    
    q_size = attn.n_heads * attn.head_dim
    query_states = qkv[1:q_size]
    gate = qkv[q_size+1:end]
    
    query_states = reshape(query_states, attn.head_dim, attn.n_heads)
    k = reshape(k, attn.head_dim, attn.n_kv)
    v = reshape(v, attn.head_dim, attn.n_kv)
    
    # Q/K normalization
    for h in 1:attn.n_heads
        q_h = view(query_states, :, h)
        Inferno.ModelCPU.rmsnorm_cpu!(q_h, q_h, attn.q_norm)
    end
    for h in 1:attn.n_kv
        k_h = view(k, :, h)
        Inferno.ModelCPU.rmsnorm_cpu!(k_h, k_h, attn.k_norm)
    end
    
    # Apply RoPE
    Inferno.ModelCPU.apply_rotary_emb!(query_states, pos, model.rope)
    Inferno.ModelCPU.apply_rotary_emb!(k, pos, model.rope)
    
    # Apply sigmoid gating
    gate_sigmoid = 1.0f0 ./ (1.0f0 .+ exp.(-gate))
    gate_reshaped = reshape(gate_sigmoid, attn.head_dim, attn.n_heads)
    query_states .*= gate_reshaped
    
    println("=== After RoPE and gating ===")
    println("query norm: ", sqrt(sum(abs2.(query_states))))
    println("k norm: ", sqrt(sum(abs2.(k))))
    
    # Update KV cache
    Inferno.ModelCPU.update_kv_cache!(cache, k, v, pos)
    
    # Attention computation
    output = zeros(Float32, attn.n_heads * attn.head_dim)
    
    gqa_ratio = div(attn.n_heads, attn.n_kv)
    seq_len = pos + 1
    
    println("\n=== Attention scores per head ===")
    
    for h in 1:attn.n_heads
        kv_h = div(h - 1, gqa_ratio) + 1
        q_h = query_states[:, h]
        
        K_h = view(cache.k, :, kv_h, 1:seq_len)
        V_h = view(cache.v, :, kv_h, 1:seq_len)
        
        # Compute scores
        scores = K_h' * q_h
        scores .*= attn.scale
        
        println("Head $h:")
        println("  q_h norm: ", sqrt(sum(abs2.(q_h))))
        println("  K_h norm: ", sqrt(sum(abs2.(K_h))))
        println("  scores: ", scores)
        
        # Softmax
        max_score = maximum(scores)
        scores_exp = exp.(scores .- max_score)
        scores_softmax = scores_exp ./ sum(scores_exp)
        
        println("  softmax: ", scores_softmax)
        
        # Weighted sum
        out_h = V_h * scores_softmax
        println("  out_h norm: ", sqrt(sum(abs2.(out_h))))
        
        output[(h-1)*attn.head_dim+1:h*attn.head_dim] .= out_h
    end
    
    println("\n=== Before wo ===")
    println("output norm: ", sqrt(sum(abs2.(output))))
    
    # Output projection
    final_output = attn.wo * output
    
    println("\n=== After wo ===")
    println("final output norm: ", sqrt(sum(abs2.(final_output))))
end

trace_attention_full()
