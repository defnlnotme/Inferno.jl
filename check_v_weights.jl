using Inferno
using LinearAlgebra

function check_v_values()
    model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    layer = model.layers[7].op
    
    # Check conv1d weights
    println("=== Conv1d weights ===")
    println("Shape: ", size(layer.ssm_conv1d))
    println("Sample weights for V channels:")
    
    # V starts after Q and K
    qk_size = layer.head_k_dim * layer.num_k_heads
    v_start = 2 * qk_size + 1
    v_end = 2 * qk_size + layer.d_inner
    
    println("QK size: ", qk_size)
    println("V channels: ", layer.d_inner)
    println("V start index: ", v_start)
    println("V end index: ", v_end)
    
    # Check the conv1d weights for V
    v_conv_weights = layer.ssm_conv1d[:, v_start:v_start+5]
    println("\nV conv1d weights sample (first 6 channels):")
    println("  min: ", minimum(v_conv_weights))
    println("  max: ", maximum(v_conv_weights))
    println("  mean: ", sum(v_conv_weights) / length(v_conv_weights))
    
    # Compare with Q conv weights
    q_conv_weights = layer.ssm_conv1d[:, 1:6]
    println("\nQ conv1d weights sample (first 6 channels):")
    println("  min: ", minimum(q_conv_weights))
    println("  max: ", maximum(q_conv_weights))
    println("  mean: ", sum(q_conv_weights) / length(q_conv_weights))
end

check_v_values()
