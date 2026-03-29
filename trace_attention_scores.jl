using Inferno
using LinearAlgebra

function trace_attention_scores()
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Prompt tokens
    tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    # Process all prompt tokens
    for (pos, tok) in enumerate(tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    
    # Now process a new token and check attention scores
    println("=== Processing token 272 after prompt ===")
    x = model.embed[:, 272]  # "\n\n"
    pos = length(tokens)  # Position 5 (0-indexed)
    
    # Manually compute attention for the last attention layer
    layer24 = model.layers[24]
    
    # Get Q projection
    qkv = layer24.op.wq * x
    q_size = layer24.op.n_heads * layer24.op.head_dim
    query = qkv[1:q_size]
    gate = qkv[q_size+1:end]
    
    # Reshape and normalize
    query = reshape(query, layer24.op.head_dim, layer24.op.n_heads)
    query = layer24.op.q_norm(query)
    
    # Apply RoPE
    Inferno.ModelCPU.apply_rotary_emb!(query, pos, model.rope)
    
    # Apply sigmoid gate
    gate = reshape(gate, layer24.op.head_dim, layer24.op.n_heads)
    gate_sigmoid = 1.0f0 ./ (1.0f0 .+ exp.(-gate))
    query .*= gate_sigmoid
    
    # Compute attention scores for head 1
    q1 = query[:, 1]
    K_h = caches[24].k[:, 1, 1:pos+1]  # (head_dim, seq_len)
    V_h = caches[24].v[:, 1, 1:pos+1]
    
    scores = K_h' * q1  # (seq_len,)
    scale = 1.0f0 / sqrt(Float32(layer24.op.head_dim))
    scores .*= scale
    
    println("Attention scores for head 1 at position $pos:")
    println("  Raw scores: ", scores)
    
    # Softmax
    max_score = maximum(scores)
    exp_scores = exp.(scores .- max_score)
    probs = exp_scores ./ sum(exp_scores)
    println("  Softmax probs: ", probs)
    
    # Check where attention is focused
    println("\n  Attention weights:")
    for (i, p) in enumerate(probs)
        if i <= length(tokens)
            println("    Position $(i-1) (\"$(Inferno.Tokenizer.decode(tokenizer, [tokens[i]]))\"): $(round(p, digits=4))")
        else
            println("    Position $(i-1): $(round(p, digits=4))")
        end
    end
end

trace_attention_scores()
