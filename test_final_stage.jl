using Inferno
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Create a test input with norm 28.3 (like after processing prompt)
x = randn(Float32, 1024)
x = x .* (28.3f0 / sqrt(sum(abs2.(x))))
println("Input norm: ", sqrt(sum(abs2.(x))))

# Apply final norm
x_normed = model.final_norm(x)
println("After final norm: ", sqrt(sum(abs2.(x_normed))))

# Apply LM head
logits = model.lm_head * x_normed
println("\nLogits:")
println("Max: ", maximum(logits))
println("Min: ", minimum(logits))
println("Norm: ", sqrt(sum(abs2.(logits))))

# Top predictions
top_indices = sortperm(logits, rev=true)[1:10]
println("\nTop 10:")
for idx in top_indices
    println("  $(idx-1): ", round(logits[idx], digits=3))
end
