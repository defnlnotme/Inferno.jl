using Inferno

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Check quantization type for each layer's MLP
println("\nMLP weight types per layer:")
for (i, layer) in enumerate(model_q.layers)
 println("Layer $(i-1): gate=$(typeof(layer.mlp.gate_weight)), up=$(typeof(layer.mlp.up_weight)), down=$(typeof(layer.mlp.down_weight))")
end
