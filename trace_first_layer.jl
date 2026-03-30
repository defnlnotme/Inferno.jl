using Inferno
using LinearAlgebra

function trace_first_layer()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Initialize caches and reset states
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token "The"
    x = model.embed[:, 761]
    println("=== First token \"The\" ===")
    println("Embedding norm: ", sqrt(sum(abs2.(x))))
    println("Embedding sample: ", x[1:5])
    
    # Apply input norm
    layer1 = model.layers[1]
    x_normed = layer1.in_norm(x)
    println("\nAfter input norm:")
    println("Norm: ", sqrt(sum(abs2.(x_normed))))
    println("Sample: ", x_normed[1:5])
    
    # Process through SSM
    ssm = layer1.op
    println("\n=== SSM Layer 1 ===")
    println("in_proj shape: ", size(ssm.in_proj))
    println("gate_proj shape: ", size(ssm.gate_proj))
    println("ssm_out shape: ", size(ssm.ssm_out))
    
    # Project to QKV
    qkv = ssm.in_proj * x_normed
    println("\nQKV projection:")
    println("Shape: ", size(qkv))
    println("Sample: ", qkv[1:5])
    
    # Apply gate
    gate = ssm.gate_proj * x_normed
    println("\nGate projection:")
    println("Shape: ", size(gate))
    println("Sample: ", gate[1:5])
    
    # Apply sigmoid to gate
    gate_sigmoid = Infra.sigmoid.(gate)
    println("\nAfter sigmoid:")
    println("Sample: ", gate_sigmoid[1:5])
end

trace_first_layer()
