using Inferno
using Statistics

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Get embedding
    h = copy(model.embed[:, 760])
    println("Input: norm=$(sqrt(sum(abs2, h))), mean=$(mean(h)), std=$(std(h))")

    # Manual RMSNorm computation
    norm = model.layers[1].in_norm

    ss = sum(abs2, h)
    m = ss / length(h)
    scale = 1.0f0 / sqrt(m + norm.eps)

    println("\nRMSNorm computation:")
    println("  ss (sum of squares): ", ss)
    println("  m (mean of squares): ", m)
    println("  rms: ", sqrt(m))
    println("  scale: ", scale)
    println("  eps: ", norm.eps)

    h_norm_manual = h .* scale .* norm.weight
    println("\nManual normalized: norm=$(sqrt(sum(abs2, h_norm_manual)))")

    # Using the model's RMSNorm
    h_norm_model = norm(h)
    println("Model normalized: norm=$(sqrt(sum(abs2, h_norm_model)))")

    # Check if they match
    println("\nDifference: ", maximum(abs.(h_norm_manual - h_norm_model)))
end

main()
