using Inferno
using Statistics

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)

# Single token forward pass test
println("\nTesting single token forward pass...")

# Get embedding for first token
tok_id = 9420  # "Hello"
x_q = model_q.embed[:, tok_id+1]

# Create caches
cache_q = ModelCPU.init_kv_cache_cpu(model_q.config, 1)
caches_q = [ModelCPU.init_kv_cache_cpu(model_q.config, 1) for _ in model_q.layers]

# Forward pass through first layer only
l1_q = model_q.layers[1]
println("\nLayer 1 type: ", l1_q.is_ssm ? "SSM" : "Attention")

# Test MLP directly
println("\nTesting MLP forward pass...")
mlp = l1_q.mlp

# Create test input
test_input = randn(Float32, model_q.config.hidden_size)

# Run through MLP
mlp_out = ModelCPU.mlp_forward(mlp, test_input)
println("MLP output (first 5): ", mlp_out[1:5])

# Load full model and compare
println("\nLoading full model for comparison...")
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Test same MLP
l1_f = model_f.layers[1]
mlp_f = l1_f.mlp

# Use same input
mlp_out_f = ModelCPU.mlp_forward(mlp_f, test_input)
println("Full MLP output (first 5): ", mlp_out_f[1:5])

# Compare
diff = mlp_out .- mlp_out_f
println("\nMLP output comparison:")
println("  Max diff: ", maximum(abs.(diff)))
println("  Mean diff: ", mean(abs.(diff)))
println("  Relative error: ", mean(abs.(diff)) / mean(abs.(mlp_out_f)) * 100, " %")
