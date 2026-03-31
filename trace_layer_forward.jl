using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset states
for layer in model.layers
    if layer.is_ssm
        Inferno.ModelCPU.reset_states_cpu!(layer.op)
    end
end

# Get "The" embedding
token = 761
x = model.embed[:, token + 1]  # +1 for 1-indexing
println("Initial embedding norm: ", round(norm(x), digits=5))
println("Expected: ~1.4189")

# Check layer 0 (SSM)
layer0 = model.layers[1]
cache0 = Inferno.ModelCPU.init_kv_cache_cpu(model.config)

println("\n=== Layer 0 Forward ===")
println("Layer type: ", layer0.is_ssm ? "SSM" : "Attention")

# Manual forward through layer 0
x_in = x
x_norm = layer0.in_norm(x_in)
println("After in_norm: ", round(norm(x_norm), digits=5))

# SSM forward
x_residual = layer0.op(x_norm, 0, model.rope, cache0)
println("After SSM: ", round(norm(x_residual), digits=5))

# Residual add
x_after = x_in + x_residual
println("After residual: ", round(norm(x_after), digits=5))

# Post norm
x_post = layer0.post_norm(x_after)
println("After post_norm: ", round(norm(x_post), digits=5))

# FFN
ffn_out = layer0.mlp(x_post)
println("After FFN: ", round(norm(ffn_out), digits=5))

# Final
x_out = x_post + ffn_out
println("Final output: ", round(norm(x_out), digits=5))

# Compare with layer call
cache0_alt = Inferno.ModelCPU.init_kv_cache_cpu(model.config)
Inferno.ModelCPU.reset_states_cpu!(layer0.op)
x_layer = layer0(x, 0, model.rope, cache0_alt)
println("\nDirect layer call: ", round(norm(x_layer), digits=5))
