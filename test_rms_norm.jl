using Inferno
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Create a test input with norm 2.272
x = randn(Float32, 1024)
x = x .* (2.272f0 / sqrt(sum(abs2.(x))))
println("Input norm: ", sqrt(sum(abs2.(x))))

# Apply RMS norm with layer 4's norm
norm_layer = model.layers[4].in_norm

# Manual RMS norm
ss = sum(abs2.(x))
m = ss / length(x)
scale = 1.0f0 / sqrt(m + norm_layer.eps)
manual_output = x .* scale .* norm_layer.weight

println("\nManual RMS norm:")
println("Sum of squares: ", ss)
println("Mean of squares: ", m)
println("Scale: ", scale)
println("Output norm: ", sqrt(sum(abs2.(manual_output))))
println("Sample output: ", manual_output[1:5])

# Apply using the norm layer
layer_output = norm_layer(x)
println("\nUsing norm layer:")
println("Output norm: ", sqrt(sum(abs2.(layer_output))))
println("Sample output: ", layer_output[1:5])
