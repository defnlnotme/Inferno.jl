using Inferno
using LinearAlgebra

function debug_attention_scores()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Process tokens
    tokens = [761, 6512]  # "The capital"
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    
    for layer in model.layers
        if layer.is_ssm
            Inferno.ModelCPU.reset_states_cpu!(layer.op)
        end
    end
    
    # Process both tokens
    for (pos, tok) in enumerate(tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    
    # Check attention scores for layer 4 (first attention layer)
    layer4 = model.layers[4]
    
    # Get the input to layer 4 for token 2
    x2 = model.embed[:, tokens[2]]
    for (i, layer) in enumerate(model.layers[1:3])
        x2 = layer(x2, 1, model.rope, caches[i])
    end
    
    # Now compute attention manually
    x_norm = layer4.in_norm(x2)
    
    # Q, K, V projections
    qkv = layer4.op.wq * x_norm
    k_new = layer4.op.wk * x_norm
    v_new = layer4.op.wv * x_norm
    
    q_size = layer4.op.n_heads * layer4.op.head_dim
    query = qkv[1:q_size]
    gate = qkv[q_size+1:end]
    
    # Reshape
    query = reshape(query, layer4.op.head_dim, layer4.op.n_heads)
    k_new = reshape(k_new, layer4.op.head_dim, layer4.op.n_kv)
    v_new = reshape(v_new, layer4.op.head_dim, layer4.op.n_kv)
    
    # Apply Q/K normalization
    for h in 1:layer4.op.n_heads
        q_h = view(query, :, h)
        Inferno.ModelCPU.rmsnorm_cpu!(q_h, q_h, layer4.op.q_norm)
    end
    for h in 1:layer4.op.n_kv
        k_h = view(k_new, :, h)
        Inferno.ModelCPU.rmsnorm_cpu!(k_h, k_h, layer4.op.k_norm)
    end
    
    # Apply RoPE
    Inferno.ModelCPU.apply_rotary_emb!(query, 1, model.rope)
    Inferno.ModelCPU.apply_rotary_emb!(k_new, 1, model.rope)
    
    # Apply SiLU gating
    gate_silu = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
    gate_reshaped = reshape(gate_silu, layer4.op.head_dim, layer4.op.n_heads)
    query .*= gate_reshaped
    
    # Compute attention scores for head 1
    q1 = query[:, 1]
    K_cache = caches[4].k[:, 1, 1:2]  # (head_dim, 2)
    
    scores = K_cache' * q1  # (2,) = (2, head_dim) * (head_dim,)
    scores .*= layer4.op.scale
    
    println("Attention scores for head 1 at layer 4:")
    println("  Raw scores: ", scores)
    
    # Softmax
    max_score = maximum(scores)
    scores_exp = exp.(scores .- max_score)
    scores_softmax = scores_exp ./ sum(scores_exp)
    println("  Softmax: ", scores_softmax)
    println("  Attention on position 1: ", scores_softmax[1])
    println("  Attention on position 2: ", scores_softmax[2])
end

debug_attention_scores()
