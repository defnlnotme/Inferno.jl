using Inferno

# Force recompilation by modifying the code
# First, let me trace exactly what's happening

function trace_loading()
    println("Loading model...")
    
    # Call load_model_cpu with explicit keep_quantized=false
    model, file = Inferno.LoaderCPU.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)
    
    mlp6 = model.layers[6].mlp
    
    println("\nLoaded gate_weight:")
    println("  Type: ", typeof(mlp6.gate_weight))
    println("  Shape: ", size(mlp6.gate_weight))
    println("  First 5 values: ", round.(mlp6.gate_weight[1, 1:5], digits=5))
end

trace_loading()
