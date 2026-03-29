using Inferno
using Printf
using Statistics
using Inferno.Tokenizer
using Inferno.ModelCPU

println("Loading model...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Get embedding for first token
tok_id = 1
x = model_q.embed[:, tok_id]
x_f = model_f.embed[:, tok_id]

println("\nFirst token embedding:")
println(" Quantized [1:5]: ", x[1:5])
println(" Float [1:5]: ", x_f[1:5])

# Create caches
caches_q = [ModelCPU.init_kv_cache_cpu(model_q.config) for _ in 1:length(model_q.layers)]
caches_f = [ModelCPU.init_kv_cache_cpu(model_f.config) for _ in 1:length(model_f.layers)]

# Process through first layer
layer_q = model_q.layers[1]
layer_f = model_f.layers[1]

println("\nProcessing through layer 1...")
y_q = layer_q(x, 0, model_q.rope, caches_q[1])
y_f = layer_f(x_f, 0, model_f.rope, caches_f[1])

println("\nLayer 1 output comparison:")
println(" Quantized [1:5]: ", y_q[1:5])
println(" Float [1:5]: ", y_f[1:5])
println(" Max diff: ", maximum(abs.(y_q .- y_f)))
println(" Mean diff: ", mean(abs.(y_q .- y_f)))

# Check if outputs are similar
if maximum(abs.(y_q .- y_f)) < 0.01
 println("\n✓ Layer 1 output matches within tolerance")
else
 println("\n✗ Layer 1 output differs significantly!")
end
