#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

# Load CPU model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

println("Layer order:")
for (i, layer) in enumerate(model.layers)
    layer_type = typeof(layer.op)
    println("  Layer $i: $layer_type")
end
