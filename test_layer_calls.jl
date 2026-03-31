using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Get "The" embedding
token = 761
x = model.embed[:, token + 1]

# Test 1: Reset state, then call layer
layer0 = model.layers[1]
cache1 = Inferno.ModelCPU.init_kv_cache_cpu(model.config)
Inferno.ModelCPU.reset_states_cpu!(layer0.op)  # Reset SSM state
x_out1 = layer0(x, 0, model.rope, cache1)
println("Test 1 (reset then call): ", round(norm(x_out1), digits=5))

# Test 2: Reset state, then call again
Inferno.ModelCPU.reset_states_cpu!(layer0.op)
cache2 = Inferno.ModelCPU.init_kv_cache_cpu(model.config)
x_out2 = layer0(x, 0, model.rope, cache2)
println("Test 2 (reset then call again): ", round(norm(x_out2), digits=5))

# Test 3: Manual forward without residual bug
Inferno.ModelCPU.reset_states_cpu!(layer0.op)
cache3 = Inferno.ModelCPU.init_kv_cache_cpu(model.config)

x_norm = layer0.in_norm(x)
x_residual = layer0.op(x_norm, 0, model.rope, cache3)
x_after = x + x_residual
x_post = layer0.post_norm(x_after)
ffn_out = layer0.mlp(x_post)
x_out3 = x_after + ffn_out  # Correct!

println("Test 3 (manual, correct): ", round(norm(x_out3), digits=5))

# Test 4: Manual forward WITH the bug
Inferno.ModelCPU.reset_states_cpu!(layer0.op)
cache4 = Inferno.ModelCPU.init_kv_cache_cpu(model.config)

x_norm = layer0.in_norm(x)
x_residual = layer0.op(x_norm, 0, model.rope, cache4)
x_after = x + x_residual
x_post = layer0.post_norm(x_after)
ffn_out = layer0.mlp(x_post)
x_out4 = x_post + ffn_out  # WRONG!

println("Test 4 (manual, wrong): ", round(norm(x_out4), digits=5))
