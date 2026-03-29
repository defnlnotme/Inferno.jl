using Inferno
using Statistics
using LinearAlgebra

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Test MLP with known input
    mlp = model.layers[1].mlp
    x = randn(Float32, 1024)
    x = x ./ std(x)  # Normalize

    println("Input: mean=$(mean(x)), std=$(std(x))")

    # Gate projection
    gate = mlp.gate_weight * x
    println("Gate (before SiLU): mean=$(mean(gate)), std=$(std(gate))")

    # SiLU activation
    gate_silu = gate ./ (1.0f0 .+ exp.(-gate))
    println("Gate (after SiLU): mean=$(mean(gate_silu)), std=$(std(gate_silu))")

    # Up projection
    up = mlp.up_weight * x
    println("Up: mean=$(mean(up)), std=$(std(up))")

    # Element-wise multiply
    hidden = gate_silu .* up
    println("Hidden (gate * up): mean=$(mean(hidden)), std=$(std(hidden))")

    # Down projection
    out = mlp.down_weight * hidden
    println("Output: mean=$(mean(out)), std=$(std(out))")

    # Check for any NaN or Inf
    println("\nAny NaN: ", any(isnan, out))
    println("Any Inf: ", any(isinf, out))
end

main()
