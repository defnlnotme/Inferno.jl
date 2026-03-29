using Inferno
using Statistics
using LinearAlgebra

function main()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Process through first layer completely
    h = copy(model.embed[:, 760])
    layer = model.layers[1]
    ssm = layer.op

    println("=== Full SSM Layer Trace ===")
    println("Input: norm=$(sqrt(sum(abs2, h)))")

    # in_norm
    h_norm = layer.in_norm(h)
    println("After in_norm: norm=$(sqrt(sum(abs2, h_norm)))")

    # SSM projections
    qkv = ssm.in_proj * h_norm
    z = ssm.gate_proj * h_norm

    println("\nSSM projections:")
    println("  qkv norm: $(sqrt(sum(abs2, qkv)))")
    println("  z norm: $(sqrt(sum(abs2, z)))")

    # Conv state update
    ssm.conv_state[:, 1:3] .= ssm.conv_state[:, 2:4]
    ssm.conv_state[:, 4] .= qkv

    # Conv1d
    x_conv = zeros(Float32, size(ssm.ssm_conv1d, 2))
    for k in 1:ssm.conv_kernel
        for c in 1:length(x_conv)
            x_conv[c] += ssm.conv_state[c, k] * ssm.ssm_conv1d[k, c]
        end
    end
    println("\nAfter conv1d: norm=$(sqrt(sum(abs2, x_conv)))")

    # SiLU
    @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))
    println("After SiLU: norm=$(sqrt(sum(abs2, x_conv)))")

    # Split Q/K/V
    qk_size = ssm.num_k_heads * ssm.head_k_dim
    q_all = x_conv[1:qk_size]
    k_all = x_conv[qk_size+1:2*qk_size]
    v_all = x_conv[2*qk_size+1:2*qk_size+ssm.d_inner]

    println("\nQ/K/V split:")
    println("  q_all norm: $(sqrt(sum(abs2, q_all)))")
    println("  k_all norm: $(sqrt(sum(abs2, k_all)))")
    println("  v_all norm: $(sqrt(sum(abs2, v_all)))")

    # Alpha/beta
    alpha_proj = ssm.ssm_alpha_weight * h_norm
    beta_proj = ssm.ssm_beta_weight * h_norm

    # Process each head
    y_all = zeros(Float32, ssm.d_inner)

    for h_idx in 1:ssm.num_v_heads
        g = ((h_idx - 1) % ssm.num_k_heads) + 1

        qg = view(q_all, (g-1)*ssm.head_k_dim+1:g*ssm.head_k_dim)
        kg = view(k_all, (g-1)*ssm.head_k_dim+1:g*ssm.head_k_dim)
        vg = view(v_all, (h_idx-1)*ssm.head_v_dim+1:h_idx*ssm.head_v_dim)

        # Q/K normalization
        q_norm = sqrt(sum(abs2, qg) + 1e-6)
        k_norm = sqrt(sum(abs2, kg) + 1e-6)
        scale = 1.0f0 / sqrt(Float32(ssm.head_k_dim))

        q_normalized = qg ./ q_norm .* scale
        k_normalized = kg ./ k_norm

        # Compute decay and beta
        alpha_val = Float64(alpha_proj[h_idx]) + Float64(ssm.ssm_dt_bias[h_idx])
        alpha_val = clamp(alpha_val, -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        softplus_alpha = clamp(softplus_alpha, -10.0, 10.0)
        decay = Float32(exp(softplus_alpha * Float64(ssm.ssm_a[h_idx])))
        decay = clamp(decay, 0.0f0, 1.0f0)

        beta_val = Float64(beta_proj[h_idx])
        beta_val = clamp(beta_val, -20.0, 20.0)
        beta = Float32(1.0 / (1.0 + exp(-beta_val)))

        # Update state (position 0, so state is zeros)
        # h_new = decay * h + beta * (k * v^T)
        # Since h = 0 at position 0:
        # h_new = beta * k * v^T
        # y = q^T * h_new = q^T * beta * k * v^T = beta * (q^T * k) * v

        # q^T * k
        qk_dot = dot(q_normalized, k_normalized)
        y_h = beta * qk_dot * vg  # Simplified

        y_all[(h_idx-1)*ssm.head_v_dim+1:h_idx*ssm.head_v_dim] .= y_h
    end

    println("\nAfter SSM recurrence: norm=$(sqrt(sum(abs2, y_all)))")

    # Apply SSM norm (per-head)
    for h_idx in 1:ssm.num_v_heads
        y_h = view(y_all, (h_idx-1)*ssm.head_v_dim+1:h_idx*ssm.head_v_dim)
        ss = sum(abs2, y_h)
        m = ss / length(y_h)
        y_h .*= 1.0f0 / sqrt(m + ssm.ssm_norm.eps) .* ssm.ssm_norm.weight
    end

    println("After SSM norm: norm=$(sqrt(sum(abs2, y_all)))")

    # Apply gate with SiLU
    @. y_all = y_all * z * (1.0f0 / (1.0f0 + exp(-z)))
    println("After gate: norm=$(sqrt(sum(abs2, y_all)))")

    # Output projection
    output = ssm.ssm_out * y_all
    println("\nFinal SSM output: norm=$(sqrt(sum(abs2, output)))")

    # This is the residual that gets added to x
    println("\nResidual norm: $(sqrt(sum(abs2, output)))")
    println("Input norm was: $(sqrt(sum(abs2, h)))")
    println("Ratio: $(sqrt(sum(abs2, output)) / sqrt(sum(abs2, h)))")
end

main()
