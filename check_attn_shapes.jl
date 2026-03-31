using Inferno
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check attention layer (layer 4, which is index 3 in 0-indexed)
layer_attn = model.layers[4]  # This should be an attention layer
println("Layer 4 is SSM: ", layer_attn.is_ssm)

if !layer_attn.is_ssm
    attn = layer_attn.op
    println("\n=== Attention Layer 3 Weight Shapes ===")
    println("wq: ", size(attn.wq))
    println("wk: ", size(attn.wk))
    println("wv: ", size(attn.wv))
    println("wo: ", size(attn.wo))
    
    println("\n=== Expected shapes ===")
    println("wq: (num_heads * head_dim, hidden) = (8 * 128, 1024) = (1024, 1024)")
    println("wk: (num_kv_heads * head_dim, hidden) = (2 * 128, 1024) = (256, 1024)")
    println("wv: (num_kv_heads * head_dim, hidden) = (2 * 128, 1024) = (256, 1024)")
    println("wo: (hidden, num_heads * head_dim) = (1024, 1024)")
    
    # Test matmul
    x = randn(Float32, 1024)
    println("\n=== Matrix Mult Test ===")
    println("wq * x: ", size(attn.wq * x))
    println("wk * x: ", size(attn.wk * x))
    println("wv * x: ", size(attn.wv * x))
end
