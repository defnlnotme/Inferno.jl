using Inferno
using LinearAlgebra

function debug_mlp_loading()
    # First, manually call the exact code in load_mlp
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    prefix = "blk.6"
    
    # Extract raw tensors
    gate_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).ffn_gate.weight")
    up_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).ffn_up.weight")
    down_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).ffn_down.weight")
    
    println("=== Raw extracted tensors ===")
    println("gate_raw: ", typeof(gate_raw), " shape: ", size(gate_raw))
    println("gate_raw[1, 1:5]: ", round.(gate_raw[1, 1:5], digits=5))
    
    # Apply the transforms in load_mlp
    gate_weight = Matrix(Float32.(gate_raw'))
    up_weight = Matrix(Float32.(up_raw'))
    down_weight = Matrix(Float32.(down_raw'))
    
    println("\n=== After transform ===")
    println("gate_weight: ", typeof(gate_weight), " shape: ", size(gate_weight))
    println("gate_weight[1, 1:5]: ", round.(gate_weight[1, 1:5], digits=5))
    
    # Now load the actual model
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    mlp6 = model.layers[6].mlp
    
    println("\n=== Actual model ===")
    println("mlp6.gate_weight: ", typeof(mlp6.gate_weight), " shape: ", size(mlp6.gate_weight))
    println("mlp6.gate_weight[1, 1:5]: ", round.(mlp6.gate_weight[1, 1:5], digits=5))
    
    # Check if they're the same
    println("\n=== Comparison ===")
    println("Same values? ", gate_weight ≈ mlp6.gate_weight)
    
    # If not, let's find where the difference is
    if !(gate_weight ≈ mlp6.gate_weight)
        println("Finding differences...")
        
        # Check if it's a transpose issue
        println("\ngate_weight'[1, 1:5]: ", round.(gate_weight'[1, 1:5], digits=5))
        println("mlp6.gate_weight'[1, 1:5]: ", round.(mlp6.gate_weight'[1, 1:5], digits=5))
        
        # Check if sizes match
        println("\nSizes: gate_weight = ", size(gate_weight), ", mlp6.gate_weight = ", size(mlp6.gate_weight))
    end
end

debug_mlp_loading()
