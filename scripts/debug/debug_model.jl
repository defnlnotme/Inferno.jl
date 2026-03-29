using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU
using Inferno.Dequant
using Statistics

# Load model
model, file = LoaderCPU.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("\n" * "="^60)
println("TEST FORWARD PASS")
println("="^60)

# Initialize
caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
ModelCPU.reset_states_cpu!(model)

# Get embedding for token 1
x_in = view(model.embed, :, 1)
println("Input embedding shape: ", size(x_in))
println("Input embedding sample: ", x_in[1:5])

# Forward through first SSM layer
x_out = model.layers[1](x_in, 0, model.rope, caches[1])
println("\nAfter layer 0 (SSM):")
println("  shape: ", size(x_out))
println("  sample: ", x_out[1:5])
println("  any NaN: ", any(isnan, x_out))
println("  mean: ", mean(x_out))
println("  std: ", std(x_out))

# Full forward pass
caches2 = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
ModelCPU.reset_states_cpu!(model)

x2 = copy(view(model.embed, :, 1))
for (i, layer) in enumerate(model.layers)
    x2 = layer(x2, 0, model.rope, caches2[i])
end
x2 = model.final_norm(x2)
logits = model.lm_head * x2

println("\nAfter full forward pass:")
println("  logits shape: ", size(logits))
println("  any NaN: ", any(isnan, logits))
println("  mean: ", mean(logits))
println("  std: ", std(logits))
println("  min: ", minimum(logits))
println("  max: ", maximum(logits))
println("  top 5 tokens: ", sortperm(logits, rev=true)[1:5])
println("  top 5 values: ", logits[sortperm(logits, rev=true)[1:5]])
