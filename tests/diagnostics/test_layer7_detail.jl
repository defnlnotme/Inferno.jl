using Inferno
using Statistics

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

    h = copy(model.embed[:, 761])
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Forward through layers 1-6
    for i in 1:6
        layer = model.layers[i]
        h = layer(h, 0, model.rope, caches[i])
    end

    println("After layer 6:")
    println("  h norm: ", round(sqrt(sum(abs2, h)), digits=3))

    # Now trace layer 7 in detail
    layer = model.layers[7]
    ssm = layer.op
    
    # Apply in_norm
    h_norm = layer.in_norm(h)
    println("\nLayer 7 after in_norm:")
    println("  norm: ", round(sqrt(sum(abs2, h_norm)), digits=3))
    
    # QKV projection
    qkv = ssm.in_proj * h_norm
    z = ssm.gate_proj * h_norm
    println("\nAfter projection:")
    println("  qkv norm: ", round(sqrt(sum(abs2, qkv)), digits=3))
    println("  z norm: ", round(sqrt(sum(abs2, z)), digits=3))
    
    # Convolution
    conv_out = zeros(Float32, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        for k in 1:ssm.conv_kernel
            conv_out[c] += ssm.ssm_conv1d[k, c] * qkv[c]
        end
    end
    println("\nAfter conv:")
    println("  conv_out norm: ", round(sqrt(sum(abs2, conv_out)), digits=3))
    
    # SiLU
    @. conv_out = conv_out * (1.0f0 / (1.0f0 + exp(-conv_out)))
    println("\nAfter SiLU:")
    println("  conv_out norm: ", round(sqrt(sum(abs2, conv_out)), digits=3))
    
    # Split Q, K, V
    qk_size = ssm.head_k_dim * ssm.num_k_heads
    q_all = reshape(view(conv_out, 1:qk_size), ssm.head_k_dim, ssm.num_k_heads)
    k_all = reshape(view(conv_out, qk_size+1:2*qk_size), ssm.head_k_dim, ssm.num_k_heads)
    v_all = reshape(view(conv_out, 2*qk_size+1:2*qk_size+ssm.d_inner), ssm.head_v_dim, ssm.num_v_heads)
    
    println("\nQ/K/V stats:")
    println("  Q norm: ", round(sqrt(sum(abs2, q_all)), digits=3))
    println("  K norm: ", round(sqrt(sum(abs2, k_all)), digits=3))
    println("  V norm: ", round(sqrt(sum(abs2, v_all)), digits=3))
end

main()
