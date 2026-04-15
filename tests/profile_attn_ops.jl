using Inferno
using LinearAlgebra

function profile_attn_ops()
    model, tok = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B/")
    
    println("=== Attention Layer Operation Breakdown ===")
    println()
    
    # Find an attention layer
    attn_layer_idx = findfirst(l -> !l.is_ssm, model.layers)
    attn_layer = model.layers[attn_layer_idx]
    attn = attn_layer.op
    
    println("Attention layer $attn_layer_idx:")
    println("  wq: ", size(attn.wq))
    println("  wk: ", size(attn.wk))
    println("  wv: ", size(attn.wv))
    println("  wo: ", size(attn.wo))
    println("  n_heads: ", attn.n_heads)
    println("  n_kv: ", attn.n_kv)
    println("  head_dim: ", attn.head_dim)
    println()
    
    prompt_tokens = Inferno.Tokenizer.encode(tok, "test")
    
    # Warm up
    Inferno.ModelCPU.reset_states_cpu!(model)
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    _ = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
    
    x = model.embed[:, prompt_tokens[end]]
    
    # Input norm
    Inferno.ModelCPU.rmsnorm_cpu!(attn_layer.norm_buf1, x, attn_layer.in_norm)
    h_norm = attn_layer.norm_buf1
    
    # Profile Q projection
    q_out = similar(attn.qkv_buf)
    times_q = Float64[]
    for i in 1:20
        t0 = time()
        mul!(q_out, attn.wq, h_norm)
        t1 = time()
        push!(times_q, t1 - t0)
    end
    println("wq: $(round(sum(times_q)/length(times_q)*1000, digits=3)) ms")
    
    # Profile K projection
    k_out = similar(attn.k_buf)
    times_k = Float64[]
    for i in 1:20
        t0 = time()
        mul!(k_out, attn.wk, h_norm)
        t1 = time()
        push!(times_k, t1 - t0)
    end
    println("wk: $(round(sum(times_k)/length(times_k)*1000, digits=3)) ms")
    
    # Profile V projection
    v_out = similar(attn.v_buf)
    times_v = Float64[]
    for i in 1:20
        t0 = time()
        mul!(v_out, attn.wv, h_norm)
        t1 = time()
        push!(times_v, t1 - t0)
    end
    println("wv: $(round(sum(times_v)/length(times_v)*1000, digits=3)) ms")
    
    # Profile O projection
    o_out = similar(attn.wo_output_buf)
    times_o = Float64[]
    for i in 1:20
        t0 = time()
        mul!(o_out, attn.wo, v_out)
        t1 = time()
        push!(times_o, t1 - t0)
    end
    println("wo: $(round(sum(times_o)/length(times_o)*1000, digits=3)) ms")
    
    println()
    total = sum(times_q) + sum(times_k) + sum(times_v) + sum(times_o)
    println("Total projections: $(round(total/length(times_q)*1000, digits=2)) ms")
end

profile_attn_ops()
