using Inferno, LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check final norm
x = Float32.(ones(1024)) * 28.0f0
x_normed = model.final_norm(x)
println("Before final norm: ", sqrt(sum(abs2.(x))))
println("After final norm: ", sqrt(sum(abs2.(x_normed))))
println("Final norm weight sample: ", model.final_norm.weight[1:5])
