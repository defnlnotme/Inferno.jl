using Inferno
using LinearAlgebra

function test_convolution()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Get first SSM layer
    ssm = model.layers[1].op
    
    # Test input
    x = model.embed[:, 761]  # "The"
    
    # Project to QKV
    qkv = ssm.in_proj * x
    
    # Update conv state (ring buffer)
    if ssm.conv_kernel > 1
        ssm.conv_state[:, 1:(ssm.conv_kernel-1)] .= ssm.conv_state[:, 2:ssm.conv_kernel]
    end
    ssm.conv_state[:, ssm.conv_kernel] .= qkv
    
    # Compute convolution
    x_conv = Vector{Float32}(undef, ssm.conv_channels)
    for c in 1:ssm.conv_channels
        x_conv[c] = dot(view(ssm.conv_state, c, :), view(ssm.ssm_conv1d, :, c))
    end
    
    # Apply SiLU
    @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))
    
    println("Convolution output shape: ", size(x_conv))
    println("Convolution output [1:5]: ", x_conv[1:5])
    println("Convolution output [6140:6144]: ", x_conv[6140:6144])
    println()
    println("Conv state sample:")
    println("  conv_state[1, :]: ", ssm.conv_state[1, :])
    println("  conv_state[2, :]: ", ssm.conv_state[2, :])
    println()
    println("SSM conv1d sample:")
    println("  ssm_conv1d[:, 1]: ", ssm.ssm_conv1d[:, 1])
    println("  ssm_conv1d[:, 2]: ", ssm.ssm_conv1d[:, 2])
end

test_convolution()
