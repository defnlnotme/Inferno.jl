using Inferno
using Statistics
using LinearAlgebra

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Get first SSM layer
    ssm = model.layers[1].op
    in_norm = model.layers[1].in_norm

    # Input: token 760 "The"
    x = copy(view(model.embed, :, 760))

    println("=== Input ===")
    println("x: mean=$(mean(x)), std=$(std(x))")

    # Step 1: Apply input norm
    h = in_norm(x)
    println("\n=== After in_norm ===")
    println("h: mean=$(mean(h)), std=$(std(h))")
    println("h[1:5]: ", h[1:5])

    # Step 2: Input projection (in_proj)
    x_proj = ssm.in_proj * h
    println("\n=== After in_proj ===")
    println("x_proj: size=$(size(x_proj)), mean=$(mean(x_proj)), std=$(std(x_proj))")

    # Step 3: Gate projection
    z = ssm.gate_proj * h
    println("\n=== After gate_proj ===")
    println("z: size=$(size(z)), mean=$(mean(z)), std=$(std(z))")

    # Step 4: Conv1d
    # Need to use the conv_state
    conv_state = zeros(Float32, ssm.conv_channels, ssm.conv_kernel)
    # Fill with current x_proj
    # Shift existing state left and add new values
    conv_state[:, 1] .= ssm.conv_state[:, 2]
    conv_state[:, 2] .= ssm.conv_state[:, 3]
    conv_state[:, 3] .= ssm.conv_state[:, 4]
    conv_state[:, 4] .= x_proj

    # Apply conv1d
    x_conv = zeros(Float32, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        for k in 1:ssm.conv_kernel
            x_conv[c] += ssm.ssm_conv1d[k, c] * conv_state[c, k]
        end
    end

    println("\n=== After conv1d ===")
    println("x_conv: mean=$(mean(x_conv)), std=$(std(x_conv))")

    # Apply SiLU
    x_silu = x_conv ./ (1.0f0 .+ exp.(-x_conv))
    println("\n=== After SiLU ===")
    println("x_silu: mean=$(mean(x_silu)), std=$(std(x_silu))")

    # Split into Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = x_silu[1:qk_size]
    k_all = x_silu[qk_size+1:2*qk_size]
    v_all = x_silu[2*qk_size+1:2*qk_size+ssm.d_inner]

    println("\n=== Q/K/V split ===")
    println("q_all: size=$(length(q_all))")
    println("k_all: size=$(length(k_all))")
    println("v_all: size=$(length(v_all))")
end

main()
