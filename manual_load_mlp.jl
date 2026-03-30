using Inferno
using LinearAlgebra

function manual_load_mlp()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    prefix = "blk.6"
    
    # This is what load_mlp does (lines 341-348)
    gate_weight = Matrix(Float32.(Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).ffn_gate.weight"))')
    up_weight = Matrix(Float32.(Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).ffn_up.weight"))')
    down_weight = Matrix(Float32.(Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).ffn_down.weight"))')
    
    println("=== Manual MLP Loading ===")
    println("gate_weight shape: ", size(gate_weight))
    println("gate_weight[1, 1:5]: ", round.(gate_weight[1, 1:5], digits=5))
    println("up_weight shape: ", size(up_weight))
    println("up_weight[1, 1:5]: ", round.(up_weight[1, 1:5], digits=5))
    println("down_weight shape: ", size(down_weight))
    println("down_weight[1, 1:5]: ", round.(down_weight[1, 1:5], digits=5))
    
    # Now load actual model
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    mlp6 = model.layers[6].mlp
    
    println("\n=== Actual Model MLP ===")
    println("gate_weight shape: ", size(mlp6.gate_weight))
    println("gate_weight[1, 1:5]: ", round.(mlp6.gate_weight[1, 1:5], digits=5))
    println("up_weight shape: ", size(mlp6.up_weight))
    println("up_weight[1, 1:5]: ", round.(mlp6.up_weight[1, 1:5], digits=5))
    println("down_weight shape: ", size(mlp6.down_weight))
    println("down_weight[1, 1:5]: ", round.(mlp6.down_weight[1, 1:5], digits=5))
    
    # Check if they match
    println("\n=== Comparison ===")
    println("gate_weight match: ", gate_weight ≈ mlp6.gate_weight)
    println("up_weight match: ", up_weight ≈ mlp6.up_weight)
    println("down_weight match: ", down_weight ≈ mlp6.down_weight)
end

manual_load_mlp()
