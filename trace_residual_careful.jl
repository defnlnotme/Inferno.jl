using Inferno
using LinearAlgebra

function trace_residual_careful()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token "The"
    x = model.embed[:, 761]
    pos = 0
    
    println("=== Initial state ===")
    println("Input x norm: ", sqrt(sum(abs2.(x))))
    println("Input x sample: ", x[1:5])
    
    # Process through first layer manually
    layer = model.layers[1]
    x_orig = copy(x)
    
    # Step 1: Input norm
    x_norm = layer.in_norm(x)
    println("\n=== After in_norm ===")
    println("x_norm norm: ", sqrt(sum(abs2.(x_norm))))
    println("x_norm sample: ", x_norm[1:5])
    
    # Step 2: SSM - we need to reset state first
    Inferno.ModelCPU.reset_states_cpu!(model)
    ssm_out = layer.op(x_norm, pos, model.rope, caches[1])
    
    println("\n=== After SSM ===")
    println("ssm_out norm: ", sqrt(sum(abs2.(ssm_out))))
    println("ssm_out sample: ", ssm_out[1:5])
    
    # Step 3: First residual
    x_after_ssm = x_orig .+ ssm_out
    println("\n=== After SSM residual (x_orig + ssm_out) ===")
    println("x_after_ssm norm: ", sqrt(sum(abs2.(x_after_ssm))))
    println("x_after_ssm sample: ", x_after_ssm[1:5])
    
    # Check the computation: x_orig + ssm_out
    manual_residual = x_orig .+ ssm_out
    println("Manual residual norm: ", sqrt(sum(abs2.(manual_residual))))
end

trace_residual_careful()
