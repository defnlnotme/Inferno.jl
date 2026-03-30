using Inferno
using LinearAlgebra

function trace_attention_layer()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token "The"
    x = model.embed[:, 761]
    pos = 0
    
    # Process through first 3 SSM layers
    for i in 1:3
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    println("=== After 3 SSM layers ===")
    println("Hidden norm: ", sqrt(sum(abs2.(x))))
    println("Sample: ", x[1:5])
    
    # Process through attention layer (layer 4)
    println("\n=== Processing Attention Layer (layer 4) ===")
    attn = model.layers[4].op
    
    println("wq shape: ", size(attn.wq))
    println("wk shape: ", size(attn.wk))
    println("wv shape: ", size(attn.wv))
    println("wo shape: ", size(attn.wo))
    
    # Apply input norm
    x_normed = model.layers[4].in_norm(x)
    println("\nAfter input norm: ", sqrt(sum(abs2.(x_normed))))
    
    # Project Q, K, V
    q = attn.wq * x_normed
    k = attn.wk * x_normed
    v = attn.wv * x_normed
    
    println("\nQ norm: ", sqrt(sum(abs2.(q))))
    println("K norm: ", sqrt(sum(abs2.(k))))
    println("V norm: ", sqrt(sum(abs2.(v))))
    
    # Apply RoPE
    freqs = model.rope(pos)
    q_rotated = Inferno.ModelCPU.apply_rope_cpu(q, freqs, model.config)
    k_rotated = Inferno.ModelCPU.apply_rope_cpu(k, freqs, model.config)
    
    println("\nAfter RoPE:")
    println("Q norm: ", sqrt(sum(abs2.(q_rotated))))
    println("K norm: ", sqrt(sum(abs2.(k_rotated))))
end

trace_attention_layer()
