using Inferno
using Statistics
using LinearAlgebra

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Test just the first token through layer 3 (attention layer)
    token = 760  # "The"
    pos = 0

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    h = copy(model.embed[:, token])
    println("=== Initial ===")
    println("Embed: mean=$(mean(h)), std=$(std(h))")

    # Process through layers 0, 1, 2 (all SSM)
    for i in 1:3
        layer = model.layers[i]
        h = layer(h, pos, model.rope, caches[i])
        println("After layer $i: mean=$(mean(h)), std=$(std(h)), max=$(maximum(abs.(h)))")
    end

    # Now process through layer 3 (attention)
    println("\n=== Processing attention layer 4 ===")
    attn_layer = model.layers[4]
    attn = attn_layer.op

    # Apply in_norm
    h_norm = attn_layer.in_norm(h)
    println("After in_norm: mean=$(mean(h_norm)), std=$(std(h_norm))")

    # QKV projection
    qkv = attn.wq * h_norm
    k = attn.wk * h_norm
    v = attn.wv * h_norm

    println("\nQKV projections:")
    println("  qkv size: ", size(qkv))
    println("  qkv mean: ", mean(qkv), " std: ", std(qkv))
    println("  k mean: ", mean(k), " std: ", std(k))
    println("  v mean: ", mean(v), " std: ", std(v))

    # Split Q and gate
    q_size = attn.n_heads * attn.head_dim
    query = qkv[1:q_size]
    gate = qkv[q_size+1:end]

    println("\nQuery/Gate:")
    println("  query mean: ", mean(query), " std: ", std(query))
    println("  gate mean: ", mean(gate), " std: ", std(gate))

    # Reshape
    query_2d = reshape(query, attn.head_dim, attn.n_heads)
    k_2d = reshape(k, attn.head_dim, attn.n_kv)
    v_2d = reshape(v, attn.head_dim, attn.n_kv)

    # Q/K norm
    query_normed = attn.q_norm(query_2d)
    k_normed = attn.k_norm(k_2d)

    println("\nAfter Q/K norm:")
    println("  query_normed mean: ", mean(query_normed), " std: ", std(query_normed))
    println("  k_normed mean: ", mean(k_normed), " std: ", std(k_normed))

    # Apply RoPE (position 0)
    # At position 0, RoPE should be identity (cos=1, sin=0)
    # So query_rope = query_normed

    # Compute attention scores for head 1
    println("\n=== Attention computation for head 1 ===")
    q_h1 = query_normed[:, 1]  # First head query
    k_h1 = k_normed[:, 1]  # First head key (only 1 KV head maps to 4 Q heads)

    println("q_h1 norm: ", sqrt(sum(abs2, q_h1)))
    println("k_h1 norm: ", sqrt(sum(abs2, k_h1)))

    # Score
    score = dot(q_h1, k_h1) * attn.scale
    println("Attention score (q.k): ", score)
    println("scale: ", attn.scale)

    # Since this is position 0, there's only 1 key in cache
    # Softmax of single value is 1.0
    # Output is just v * 1.0 = v

    println("\nSince pos=0, attention output = v[1] = ", v_2d[1:5, 1])

    # Apply gate
    gate_2d = reshape(gate, attn.head_dim, attn.n_heads)
    gate_h1 = gate_2d[:, 1]
    println("gate_h1[1:5]: ", gate_h1[1:5])
    println("sigmoid(gate_h1[1:5]): ", 1.0f0 ./ (1.0f0 .+ exp.(-gate_h1[1:5])))
end

main()
