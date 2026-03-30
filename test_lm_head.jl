using Inferno
using LinearAlgebra

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Create a test hidden state
x = Float32.(ones(1024))
x = x ./ sqrt(sum(abs2.(x)))  # Normalize

println("Input hidden state norm: ", sqrt(sum(abs2.(x))))

# Apply LM head
logits = model.lm_head * x

println("\nLogits:")
println("  Shape: ", size(logits))
println("  Max: ", maximum(logits))
println("  Min: ", minimum(logits))
println("  Mean: ", sum(logits) / length(logits))
println("  Argmax: ", argmax(logits) - 1, " (0-indexed)")

# Compare with manual computation
# logits[i] = sum_j lm_head[i,j] * x[j]
# For token 272:
logits_272 = dot(model.lm_head[273, :], x)
println("\nManual logits[272]: ", logits_272)
println("Computed logits[272]: ", logits[273])

# Check if the LM head is transposed
# If lm_head is stored as (hidden, vocab), then we need to do x * lm_head
println("\n=== Checking if LM head needs transpose ===")
logits_alt = x' * model.lm_head
println("Alternative (x' * lm_head):")
println("  Shape: ", size(logits_alt))
println("  Max: ", maximum(logits_alt))
println("  Argmax: ", argmax(logits_alt) - 1, " (0-indexed)")
