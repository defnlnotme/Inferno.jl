using Inferno
using LinearAlgebra

function check_qkv_raw()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Process first token through layers 1-6
    tok = 761
    pos = 0
    x = model.embed[:, tok]
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    layer = model.layers[7]
    ssm = layer.op
    x_norm = layer.in_norm(x)
    
    # Raw qkv before convolution
    qkv = ssm.in_proj * x_norm
    
    println("=== Raw qkv (before convolution) ===")
    println("Total norm: ", round(sqrt(sum(abs2.(qkv))), digits=3))
    
    # Split into Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    
    Q_raw = qkv[1:qk_size]
    K_raw = qkv[qk_size+1:2*qk_size]
    V_raw = qkv[2*qk_size+1:end]
    
    println("\nQ_raw norm: ", round(sqrt(sum(abs2.(Q_raw))), digits=3))
    println("Q_raw sample: ", Q_raw[1:5])
    
    println("\nK_raw norm: ", round(sqrt(sum(abs2.(K_raw))), digits=3))
    println("K_raw sample: ", K_raw[1:5])
    
    println("\nV_raw norm: ", round(sqrt(sum(abs2.(V_raw))), digits=3))
    println("V_raw sample: ", V_raw[1:5])
    
    # After convolution
    if ssm.conv_kernel > 1
        ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
    end
    ssm.conv_state[:, ssm.conv_kernel] .= qkv
    
    x_conv = Vector{Float32}(undef, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
    end
    
    println("\n=== After convolution (before SiLU) ===")
    println("x_conv norm: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    
    Q_conv = x_conv[1:qk_size]
    K_conv = x_conv[qk_size+1:2*qk_size]
    V_conv = x_conv[2*qk_size+1:end]
    
    println("\nQ_conv norm: ", round(sqrt(sum(abs2.(Q_conv))), digits=3))
    println("Q_conv sample: ", Q_conv[1:5])
    
    println("\nK_conv norm: ", round(sqrt(sum(abs2.(K_conv))), digits=3))
    println("K_conv sample: ", K_conv[1:5])
    
    println("\nV_conv norm: ", round(sqrt(sum(abs2.(V_conv))), digits=3))
    println("V_conv sample: ", V_conv[1:5])
end

check_qkv_raw()
