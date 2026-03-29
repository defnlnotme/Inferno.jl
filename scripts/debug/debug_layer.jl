using Inferno
using Statistics

function main()
    println("Loading model...")
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    println("Model loaded.\n")

    # Check the first layer (SSM)
    ssm = model.layers[1].op
    println("=== SSM Layer 0 Analysis ===")
    println("Weights loaded:")
    println("  in_proj: ", size(ssm.in_proj), " sample: ", ssm.in_proj[1:3, 1:3])
    println("  gate_proj: ", size(ssm.gate_proj), " sample: ", ssm.gate_proj[1:3, 1:3])
    println("  ssm_conv1d: ", size(ssm.ssm_conv1d), " sample: ", ssm.ssm_conv1d[1:3, 1:3])
    println("  ssm_a: ", ssm.ssm_a)
    println("  ssm_dt_bias: ", ssm.ssm_dt_bias)
    
    # Initialize
    ModelCPU.reset_states_cpu!(model)
    
    # Get embedding for a known token - token 15 is "0"
    println("\n=== Forward Pass Analysis ===")
    x = copy(view(model.embed, :, 15))  # Token "0"
    println("Input embedding (token 15 = '0'):")
    println("  mean: ", mean(x), " std: ", std(x), " min: ", minimum(x), " max: ", maximum(x))
    
    # Step through first SSM layer manually
    println("\n--- Layer 0 (SSM) ---")
    
    # 1. Input projections
    qkv = ssm.in_proj * x
    z = ssm.gate_proj * x
    println("After in_proj: mean=", mean(qkv), " std=", std(qkv))
    println("After gate_proj: mean=", mean(z), " std=", std(z))
    
    # 2. Conv operation
    println("\nConv state before: ", ssm.conv_state[1:5, 1])
    # Update conv state
    if ssm.conv_kernel > 1
        ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
    end
    ssm.conv_state[:, ssm.conv_kernel] .= qkv
    
    # Compute convolution
    x_conv = zeros(Float32, ssm.conv_channels)
    for k in 1:ssm.conv_kernel
        for c in 1:ssm.conv_channels
            x_conv[c] += ssm.conv_state[c, k] * ssm.ssm_conv1d[c, k]
        end
    end
    println("After conv: mean=", mean(x_conv), " std=", std(x_conv))
    
    # 3. SiLU activation
    @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))
    println("After SiLU: mean=", mean(x_conv), " std=", std(x_conv))
    
    # 4. Split into Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    println("\nQ shape: ", size(q_all), " K shape: ", size(k_all), " V shape: ", size(v_all))
    println("Q sample: ", q_all[1:3, 1])
    println("K sample: ", k_all[1:3, 1])
    println("V sample: ", v_all[1:3, 1])
    
    # 5. Alpha/beta projections
    alpha_proj = ssm.ssm_alpha_weight * x
    beta_proj = ssm.ssm_beta_weight * x
    println("\nalpha_proj: ", alpha_proj)
    println("beta_proj: ", beta_proj)
    
    # Now compare with full layer call
    println("\n=== Full Layer 0 Call ===")
    ModelCPU.reset_states_cpu!(model)
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    
    x2 = copy(view(model.embed, :, 15))
    y = model.layers[1](x2, 0, model.rope, caches[1])
    println("Output: mean=", mean(y), " std=", std(y), " min=", minimum(y), " max=", maximum(y))
end

main()
