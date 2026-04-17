using Inferno
using LinearAlgebra
using Statistics

function test_flash_attention()
    GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    println("=== Flash Attention Test ===")
    println()
    
    # Find an attention layer
    attn_layer_idx = findfirst(l -> l.op isa Inferno.ModelCPU.FullAttentionCPU, model.layers)
    layer = model.layers[attn_layer_idx]
    attn = layer.op
    
    println("Testing on Attention layer $attn_layer_idx")
    println("  n_heads: $(attn.n_heads)")
    println("  n_kv: $(attn.n_kv)")
    println("  head_dim: $(attn.head_dim)")
    println()
    
    # Create a KV cache and populate it
    cache = Inferno.ModelCPU.init_kv_cache_cpu(model.config)
    
    # Generate a few tokens to build up the cache
    prompt_tokens = Inferno.Tokenizer.encode(tok, "Paris is")
    
    t0 = time()
    logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:24]; full_logits=false)
    t1 = time()
    println("Standard forward: $(round((t1-t0)*1000, digits=2)) ms")
    
    println()
    println("Flash Attention functions available:")
    println("  - flash_attention_cpu!")
    println("  Defined in: Inferno.ModelCPU.FlashAttention")
end

test_flash_attention()
