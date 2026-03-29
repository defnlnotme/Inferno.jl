using Inferno
using Random
using Statistics

# Load both models
println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Get embedding
input_id = 9420  # "Hello"
x_q = model_q.embed[:, input_id+1]
x_f = model_f.embed[:, input_id+1]

println("\nEmbedding comparison:")
println("  Embedding match: ", x_q == x_f)
println("  Max diff: ", maximum(abs.(x_q .- x_f)))

# Test just the MLP of layer 1
println("\nTesting Layer 1 MLP...")
mlp_q = model_q.layers[1].mlp
mlp_f = model_f.layers[1].mlp

# Use the embedding as input
out_q = ModelCPU.mlp_forward(mlp_q, x_q)
out_f = ModelCPU.mlp_forward(mlp_f, x_f)

println("  MLP output (quantized) [1:5]: ", out_q[1:5])
println("  MLP output (float) [1:5]: ", out_f[1:5])
println("  Max diff: ", maximum(abs.(out_q .- out_f)))
println("  Relative error: ", mean(abs.(out_q .- out_f)) / mean(abs.(out_f)) * 100, " %")

# Check what type layer 1 is
println("\nLayer 1 is SSM: ", model_q.layers[1].is_ssm)

# If it's SSM, the MLP output goes through different processing
# Let me check the layer types
println("\nLayer types:")
for (i, layer) in enumerate(model_q.layers)
 println("  Layer $i: ", layer.is_ssm ? "SSM" : "Attention")
end
