using Inferno
using Statistics

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Get embedding
input_id = 9420  # "Hello"
x_q = copy(model_q.embed[:, input_id+1])
x_f = copy(model_f.embed[:, input_id+1])

println("\nEmbedding match: ", x_q == x_f)

# Process through first layer manually
layer_q = model_q.layers[1]
layer_f = model_f.layers[1]

println("\nLayer 1 type: ", layer_q.is_ssm ? "SSM" : "Attention")

# Initialize caches
cache_q = ModelCPU.init_kv_cache_cpu(model_q.config, 10)
cache_f = ModelCPU.init_kv_cache_cpu(model_f.config, 10)

# Apply input norm
x_norm_q = layer_q.in_norm(x_q)
x_norm_f = layer_f.in_norm(x_f)

println("\nAfter input norm:")
println("  Quantized [1:5]: ", x_norm_q[1:5])
println("  Float [1:5]: ", x_norm_f[1:5])
println("  Max diff: ", maximum(abs.(x_norm_q .- x_norm_f)))

# Apply SSM/Attention
if layer_q.is_ssm
 println("\nApplying SSM...")
 # SSM forward
 ssm_out_q = layer_q.op(x_norm_q, 0, model_q.rope, cache_q)
 ssm_out_f = layer_f.op(x_norm_f, 0, model_f.rope, cache_f)
 
 println("  SSM output (quantized) [1:5]: ", ssm_out_q[1:5])
 println("  SSM output (float) [1:5]: ", ssm_out_f[1:5])
 println("  Max diff: ", maximum(abs.(ssm_out_q .- ssm_out_f)))
 
 # Apply residual connection
 x_q = x_q .+ ssm_out_q
 x_f = x_f .+ ssm_out_f
 
 println("\nAfter SSM residual:")
 println("  Quantized [1:5]: ", x_q[1:5])
 println("  Float [1:5]: ", x_f[1:5])
 println("  Max diff: ", maximum(abs.(x_q .- x_f)))
end

# Apply post norm
x_post_norm_q = layer_q.post_norm(x_q)
x_post_norm_f = layer_f.post_norm(x_f)

println("\nAfter post norm:")
println("  Quantized [1:5]: ", x_post_norm_q[1:5])
println("  Float [1:5]: ", x_post_norm_f[1:5])
println("  Max diff: ", maximum(abs.(x_post_norm_q .- x_post_norm_f)))

# Apply MLP
mlp_out_q = ModelCPU.mlp_forward(layer_q.mlp, x_post_norm_q)
mlp_out_f = ModelCPU.mlp_forward(layer_f.mlp, x_post_norm_f)

println("\nAfter MLP:")
println("  Quantized [1:5]: ", mlp_out_q[1:5])
println("  Float [1:5]: ", mlp_out_f[1:5])
println("  Max diff: ", maximum(abs.(mlp_out_q .- mlp_out_f)))

# Final residual
x_q = x_q .+ mlp_out_q
x_f = x_f .+ mlp_out_f

println("\nAfter MLP residual (end of layer 1):")
println("  Quantized [1:5]: ", x_q[1:5])
println("  Float [1:5]: ", x_f[1:5])
println("  Max diff: ", maximum(abs.(x_q .- x_f)))
println("  Relative error: ", mean(abs.(x_q .- x_f)) / mean(abs.(x_f)) * 100, " %")
