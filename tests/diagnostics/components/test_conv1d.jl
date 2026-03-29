#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
layer = model.layers[1]
ssm = layer.op

# Reset state
Inferno.ModelCPU.reset_states_cpu!(model)

x = model.embed[:, 562]
x_normed = layer.in_norm(x)
qkv = ssm.in_proj * x_normed

# Manual conv computation at position 0
# The implementation stores qkv at position kernel, then convolves
x_conv_manual = zeros(Float32, ssm.conv_channels)
for c in 1:ssm.conv_channels
    # At position 0, only the kernel position (4) has data
    x_conv_manual[c] = qkv[c] * ssm.ssm_conv1d[4, c]
end

println("Manual x_conv at position 0:")
println("  norm: ", norm(x_conv_manual))
println("  first 5: ", x_conv_manual[1:5])

# Now let's actually run the SSM and compare
Inferno.ModelCPU.reset_states_cpu!(model)
config = model.config
caches = [Inferno.ModelCPU.init_kv_cache_cpu(config, 512) for _ in 1:config.num_hidden_layers]

# Run the full SSM forward pass
ssm_out = ssm(x_normed, 0, model.rope, caches[1])
println("\nSSM output:")
println("  norm: ", norm(ssm_out))
println("  first 5: ", ssm_out[1:5])

# The SSM output should be non-zero since we're applying the conv1d
