using Inferno
using LinearAlgebra

# Load model
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Test with a simple prompt
prompt = "The"
tokens = Inferno.Tokenizer.encode(tok, prompt)
println("Prompt: '$prompt'")
println("Tokens: ", tokens)

# Get embedding
token_id = tokens[1] + 1  # 1-indexed
global x = model.embed[:, token_id]
println("\n=== Embedding ===")
println("Token embedding norm: ", round(norm(x), digits=5))
println("First 5 values: ", round.(x[1:5], digits=5))

# Initialize KV caches
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

# Track through each layer
println("\n=== Layer-by-Layer Trace ===")
for (i, layer) in enumerate(model.layers)
    layer_type = layer.is_ssm ? "SSM" : "Attention"
    
    # Input norm
    x_norm = layer.in_norm(x)
    
    # Run the layer operation (SSM or Attention)
    x_residual = layer.op(x_norm, i-1, model.rope, caches[i])
    
    # Add residual
    x_after_residual = x + x_residual
    
    # Post attention norm
    x_post = layer.post_norm(x_after_residual)
    
    # FFN
    ffn_out = layer.mlp(x_post)
    
    # Final output
    global x = x_post + ffn_out
    
    println("Layer $i ($layer_type):")
    println("  x_norm norm: ", round(norm(x_norm), digits=5))
    println("  x_residual norm: ", round(norm(x_residual), digits=5))
    println("  x_after_residual norm: ", round(norm(x_after_residual), digits=5))
    println("  x_post norm: ", round(norm(x_post), digits=5))
    println("  ffn_out norm: ", round(norm(ffn_out), digits=5))
    println("  output x norm: ", round(norm(x), digits=5))
end

# Final norm
x_final = model.final_norm(x)
println("\n=== Final ===")
println("After final_norm: ", round(norm(x_final), digits=5))

# LM head
logits = model.lm_head * x_final
println("Logits norm: ", round(norm(logits), digits=5))
println("Logits shape: ", size(logits))

# Top tokens
top_k = 10
top_indices = sortperm(logits, rev=true)[1:top_k]
println("\nTop $top_k tokens:")
for idx in top_indices
    token = Inferno.Tokenizer.decode(tok, [idx])
    println("  $idx: ", round(logits[idx], digits=3), " -> '$token'")
end
