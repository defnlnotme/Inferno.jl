using Inferno
using LinearAlgebra

function trace_delta_net_state()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    tok = 761  # "The"
    pos = 0
    x = model.embed[:, tok]
    
    # Process through layers 1-6
    for i in 1:6
        x = model.layers[i](x, pos, model.rope, caches[i])
    end
    
    println("=== Layer 7 SSM State Analysis ===")
    
    layer = model.layers[7]
    ssm = layer.op
    x_norm = layer.in_norm(x)
    
    # Get projections
    qkv = ssm.in_proj * x_norm
    z = ssm.gate_proj * x_norm
    
    println("qkv norm: ", round(sqrt(sum(abs2.(qkv))), digits=3))
    println("z norm: ", round(sqrt(sum(abs2.(z))), digits=3))
    
    # Check conv_state BEFORE update
    println("\nconv_state before update:")
    println("  norm: ", round(sqrt(sum(abs2.(ssm.conv_state))), digits=6))
    
    # Update conv state
    if ssm.conv_kernel > 1
        ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
    end
    ssm.conv_state[:, ssm.conv_kernel] .= qkv
    
    println("conv_state after update:")
    println("  norm: ", round(sqrt(sum(abs2.(ssm.conv_state))), digits=3))
    
    # Compute convolution
    x_conv = Vector{Float32}(undef, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
    end
    
    println("\nx_conv before SiLU:")
    println("  norm: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    println("  V part norm: ", round(sqrt(sum(abs2.(x_conv[4097:end]))), digits=3))
    
    # SiLU
    @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))
    
    println("x_conv after SiLU:")
    println("  norm: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    println("  V part norm: ", round(sqrt(sum(abs2.(x_conv[4097:end]))), digits=3))
    
    # The key insight: for first token, conv_state is all zeros except position 4
    # So x_conv[c] = qkv[c] * ssm_conv1d[4, c]
    # Let's compute this directly
    println("\n=== Direct computation for first token ===")
    println("x_conv[c] = qkv[c] * ssm_conv1d[4, c]")
    
    # Check the relationship
    x_conv_direct = qkv .* ssm.ssm_conv1d[4, :]
    println("x_conv_direct norm: ", round(sqrt(sum(abs2.(x_conv_direct))), digits=3))
    println("x_conv computed norm: ", round(sqrt(sum(abs2.(x_conv))), digits=3))
    
    # Check if they match (before SiLU)
    x_conv_before_silu = Vector{Float32}(undef, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        x_conv_before_silu[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
    end
    
    diff = x_conv_direct - x_conv_before_silu
    println("\nDifference between direct and computed: ", round(sqrt(sum(abs2.(diff))), digits=6))
end

trace_delta_net_state()
