using Inferno

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get first SSM layer
ssm_layer = model.layers[1].op

println("Checking decay computation for SSM layer:")
println("  ssm_a (first 5): ", ssm_layer.ssm_a[1:5])
println("  ssm_dt_bias (first 5): ", ssm_layer.ssm_dt_bias[1:5])

# Sample alpha_proj value (we need to compute it from input)
# For now, let's check what happens with a typical alpha value
alpha_proj = 0.5  # hypothetical value

println("\nDecay computation for head 1:")
println("  alpha_proj[1] = $alpha_proj")
println("  ssm_dt_bias[1] = ", ssm_layer.ssm_dt_bias[1])
println("  ssm_a[1] = ", ssm_layer.ssm_a[1])

alpha_val = alpha_proj + ssm_layer.ssm_dt_bias[1]
softplus_alpha = log(1.0 + exp(alpha_val))
decay = exp(softplus_alpha * ssm_layer.ssm_a[1])

println("  alpha_val = $alpha_val")
println("  softplus_alpha = $softplus_alpha")
println("  decay = exp($softplus_alpha * $(ssm_layer.ssm_a[1])) = $decay")
