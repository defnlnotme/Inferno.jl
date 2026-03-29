using Inferno
using Statistics

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

input_id = 9420  # "Hello"
x_q = copy(model_q.embed[:, input_id+1])
x_f = copy(model_f.embed[:, input_id+1])

# Process layer 0
cache_q = ModelCPU.init_kv_cache_cpu(model_q.config, 10)
cache_f = ModelCPU.init_kv_cache_cpu(model_f.config, 10)

println("\n=== Layer 0 ===")
x_q = model_q.layers[1](x_q, 0, model_q.rope, cache_q)
x_f = model_f.layers[1](x_f, 0, model_f.rope, cache_f)
println("After layer 0: max_diff=$(maximum(abs.(x_q .- x_f)))")

# Process layer 1
cache_q = ModelCPU.init_kv_cache_cpu(model_q.config, 10)
cache_f = ModelCPU.init_kv_cache_cpu(model_f.config, 10)

println("\n=== Layer 1 ===")
println("Input to layer 1 (quantized) [1:5]: ", x_q[1:5])
println("Input to layer 1 (float) [1:5]: ", x_f[1:5])

# Debug: Let's trace through layer 1 manually
layer_q = model_q.layers[2]  # Layer 1 (1-indexed)
layer_f = model_f.layers[2]

println("\nLayer 1 weights:")
println("  in_norm weight [1:5]: ", layer_q.in_norm.weight[1:5])
println("  in_norm weight match: ", layer_q.in_norm.weight == layer_f.in_norm.weight)

# Apply input norm
x_norm_q = layer_q.in_norm(x_q)
x_norm_f = layer_f.in_norm(x_f)
println("\nAfter in_norm:")
println("  quantized [1:5]: ", x_norm_q[1:5])
println("  float [1:5]: ", x_norm_f[1:5])
println("  max_diff: ", maximum(abs.(x_norm_q .- x_norm_f)))

# SSM operation
ssm_q = layer_q.op(x_norm_q, 0, model_q.rope, cache_q)
ssm_f = layer_f.op(x_norm_f, 0, model_f.rope, cache_f)
println("\nAfter SSM:")
println("  quantized [1:5]: ", ssm_q[1:5])
println("  float [1:5]: ", ssm_f[1:5])
println("  max_diff: ", maximum(abs.(ssm_q .- ssm_f)))

# After residual
x_after_ssm_q = x_q .+ ssm_q
x_after_ssm_f = x_f .+ ssm_f
println("\nAfter SSM residual:")
println("  quantized [1:5]: ", x_after_ssm_q[1:5])
println("  float [1:5]: ", x_after_ssm_f[1:5])
println("  max_diff: ", maximum(abs.(x_after_ssm_q .- x_after_ssm_f)))

# Post norm
x_post_norm_q = layer_q.post_norm(x_after_ssm_q)
x_post_norm_f = layer_q.post_norm(x_after_ssm_f)
println("\nAfter post_norm:")
println("  quantized [1:5]: ", x_post_norm_q[1:5])
println("  float [1:5]: ", x_post_norm_f[1:5])
println("  max_diff: ", maximum(abs.(x_post_norm_q .- x_post_norm_f)))

# MLP
mlp_q = ModelCPU.mlp_forward(layer_q.mlp, x_post_norm_q)
mlp_f = ModelCPU.mlp_forward(layer_f.mlp, x_post_norm_f)
println("\nAfter MLP:")
println("  quantized [1:5]: ", mlp_q[1:5])
println("  float [1:5]: ", mlp_f[1:5])
println("  max_diff: ", maximum(abs.(mlp_q .- mlp_f)))

# Final output
x_out_q = x_after_ssm_q .+ mlp_q
x_out_f = x_after_ssm_f .+ mlp_f
println("\nFinal output:")
println("  quantized [1:5]: ", x_out_q[1:5])
println("  float [1:5]: ", x_out_f[1:5])
println("  max_diff: ", maximum(abs.(x_out_q .- x_out_f)))
