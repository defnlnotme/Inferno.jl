using Inferno
using LinearAlgebra

# Load model
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Test with "The"
token = 761  # "The" token
global x = model.embed[:, token + 1]  # +1 for 1-indexing
println("Token embedding norm: ", round(norm(x), digits=5))

# Run through just the first layer to see what's happening
layer = model.layers[1]
cache = Inferno.ModelCPU.init_kv_cache_cpu(model.config)

# Input normalization
x_norm = layer.in_norm(x)
println("\nAfter in_norm:")
println("  x_norm norm: ", round(norm(x_norm), digits=5))
println("  First 5 values: ", round.(x_norm[1:5], digits=5))

# SSM layer
println("\nSSM layer weights:")
ssm = layer.op
println("  in_proj shape: ", size(ssm.in_proj))
println("  gate_proj shape: ", size(ssm.gate_proj))
println("  ssm_out shape: ", size(ssm.ssm_out))

# Run SSM
x_residual = layer.op(x_norm, 0, model.rope, cache)
println("\nAfter SSM:")
println("  x_residual norm: ", round(norm(x_residual), digits=5))
println("  First 5 values: ", round.(x_residual[1:5], digits=5))

# Check if there are NaN or Inf values
if any(isnan.(x_residual)) || any(isinf.(x_residual))
    println("  WARNING: NaN or Inf values detected!")
end

# Add residual
x_after = x + x_residual
println("\nAfter adding residual:")
println("  x_after norm: ", round(norm(x_after), digits=5))

# Post norm
x_post = layer.post_norm(x_after)
println("\nAfter post_norm:")
println("  x_post norm: ", round(norm(x_post), digits=5))

# FFN
ffn_out = layer.mlp(x_post)
println("\nAfter FFN:")
println("  ffn_out norm: ", round(norm(ffn_out), digits=5))
println("  First 5 values: ", round.(ffn_out[1:5], digits=5))

# Final
global x = x_post + ffn_out
println("\nFinal layer output:")
println("  x norm: ", round(norm(x), digits=5))
