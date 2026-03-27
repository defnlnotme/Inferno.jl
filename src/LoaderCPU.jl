"""
CPU-only model loader for Inferno.jl
Loads GGUF models without GPU dependencies.
"""
module LoaderCPU

using ..ModelCPU
using ..GGUF
using ..Dequant

export load_model_cpu

function extract_tensor_cpu(file::GGUF.GGUFFile, info::GGUF.TensorInfo)
    num_elements = Int(prod(info.dimensions))
    start = Int(file.data_offset + info.offset) + 1

    data = if info.type == GGUF.GGML_TYPE_F32
        reinterpret(Float32, @view file.tensor_data[start:start+num_elements*4-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_F16
        reinterpret(Float16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_BF16
        raw_u16 = reinterpret(UInt16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect
        reinterpret(Float32, UInt32.(raw_u16) .<< 16)
    elseif info.type == GGUF.GGML_TYPE_Q5_K
        Dequant.dequantize_q5_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q4_K
        Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q8_0
        Dequant.dequantize_q8_0(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_XS
        Dequant.dequantize_iq2_xs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_S
        Dequant.dequantize_iq2_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_XXS
        Dequant.dequantize_iq3_xxs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_S
        Dequant.dequantize_iq3_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ4_XS
        Dequant.dequantize_iq4_xs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q2_K
        Dequant.dequantize_q2_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q3_K
        Dequant.dequantize_q3_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q6_K
        Dequant.dequantize_q6_k(@view(file.tensor_data[start:end]), num_elements)
    else
        @warn "Unhandled type, zeroing" type = info.type
        zeros(Float32, num_elements)
    end

    dims = Tuple(Int.(info.dimensions))
    if length(dims) > 2
        return reshape(data, dims)
    else
        inner = dims[1]
        outer = length(dims) > 1 ? dims[2] : 1
        return reshape(data, inner, outer)
    end
end

function extract_tensor_cpu(file::GGUF.GGUFFile, name::String)
    if !haskey(file.tensors, name)
        @warn "Tensor not found, using zero surrogate" name = name
        return zeros(Float32, 1, 1)
    end
    info = file.tensors[name]
    return extract_tensor_cpu(file, info)
end

function load_model_cpu(path::String)
    # Load GGUF file
    file = GGUF.read_gguf(path)
    
    # Get config
    config = get_config(file)
    
    println("Config: hidden=$(config.hidden_size), layers=$(config.num_hidden_layers), heads=$(config.num_attention_heads)")
    
    # Load embedding
    embed = extract_tensor_cpu(file, "token_embd.weight")
    embed = Float32.(embed)
    
    println("Embedding: $(size(embed))")
    
    # Load layers
    layers = ModelCPU.DecoderLayerCPU[]
    
    for i in 0:(config.num_hidden_layers - 1)
        layer = load_layer(file, i, config)
        push!(layers, layer)
        println("  Layer $i: $(layer.is_ssm ? "SSM" : "Attention")")
    end
    
    # Load final norm
    final_norm_w = Float32.(extract_tensor_cpu(file, "output_norm.weight"))
    final_norm = ModelCPU.RMSNormCPU(final_norm_w, config.rms_norm_eps)
    
    # LM head (check if tied or separate)
    if haskey(file.tensors, "output.weight")
        lm_head = Float32.(extract_tensor_cpu(file, "output.weight"))
    else
        lm_head = embed'
    end
    
    # Create RoPE
    rope = ModelCPU.RotaryEmbeddingCPU(config.head_dim, config.rope_theta, config.max_position_embeddings)
    
    return ModelCPU.QwenModelCPU(config, embed, lm_head, layers, final_norm, rope), file
end

function get_config(file::GGUF.GGUFFile)
    arch = get(file.metadata, "general.architecture", "qwen")
    
    config = ModelCPU.QwenConfigCPU(
        architecture = Symbol(arch),
        vocab_size = get(file.metadata, "qwen3.vocab_size", get(file.metadata, "vocab_size", 151936)),
        hidden_size = get(file.metadata, "qwen3.embedding_length", get(file.metadata, "embedding_length", 1024)),
        intermediate_size = get(file.metadata, "qwen3.feed_forward_length", get(file.metadata, "feed_forward_length", 3584)),
        num_hidden_layers = get(file.metadata, "qwen3.block_count", get(file.metadata, "block_count", 24)),
        num_attention_heads = get(file.metadata, "qwen3.attention.head_count", get(file.metadata, "attention.head_count", 8)),
        num_key_value_heads = get(file.metadata, "qwen3.attention.head_count_kv", get(file.metadata, "attention.head_count_kv", 2)),
        head_dim = get(file.metadata, "qwen3.attention.key_length", get(file.metadata, "attention.key_length", 256)),
        rms_norm_eps = Float32(get(file.metadata, "qwen3.attention.layer_norm_rms_epsilon", get(file.metadata, "attention.layer_norm_rms_epsilon", 1e-6))),
        rope_theta = Float32(get(file.metadata, "qwen3.rope.freq_base", get(file.metadata, "rope.freq_base", 10000000.0))),
        max_position_embeddings = get(file.metadata, "qwen3.context_length", get(file.metadata, "context_length", 4096)),
    )
    
    return config
end

function load_layer(file::GGUF.GGUFFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU)
    prefix = "blk.$(layer_idx)"
    
    # Load norms
    in_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).attn_norm.weight"))
    in_norm = ModelCPU.RMSNormCPU(in_norm_w, config.rms_norm_eps)
    
    post_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).post_attention_norm.weight"))
    post_norm = ModelCPU.RMSNormCPU(post_norm_w, config.rms_norm_eps)
    
    # Check if SSM layer - it has ssm_a tensor
    is_ssm = haskey(file.tensors, "$(prefix).ssm_a")
    
    if is_ssm
        op = load_ssm_layer(file, layer_idx, config)
    else
        op = load_attention_layer(file, layer_idx, config)
    end
    
    # Load MLP
    mlp = load_mlp(file, layer_idx, config)
    
    return ModelCPU.DecoderLayerCPU(in_norm, op, post_norm, mlp, is_ssm)
end

function load_ssm_layer(file::GGUF.GGUFFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU)
    prefix = "blk.$(layer_idx)"
    
    # Load weights - note: GGUF stores as (out_features, in_features), we need to transpose
    # attn_qkv.weight is the combined Q,K,V projection (also used as input projection for SSM)
    in_proj = Float32.(extract_tensor_cpu(file, "$(prefix).attn_qkv.weight"))'
    gate_proj = Float32.(extract_tensor_cpu(file, "$(prefix).attn_gate.weight"))'
    ssm_out = Float32.(extract_tensor_cpu(file, "$(prefix).ssm_out.weight"))'
    
    # Conv1d - stored as (out_features, kernel_size)
    ssm_conv1d = Float32.(extract_tensor_cpu(file, "$(prefix).ssm_conv1d.weight"))'
    
    # Alpha/beta weights
    ssm_alpha_weight = Float32.(extract_tensor_cpu(file, "$(prefix).ssm_alpha.weight"))'
    ssm_beta_weight = Float32.(extract_tensor_cpu(file, "$(prefix).ssm_beta.weight"))'
    
    # SSM parameters - ssm_a is a tensor, not a bias
    ssm_a = Float32.(vec(extract_tensor_cpu(file, "$(prefix).ssm_a")))
    ssm_dt_bias = Float32.(vec(extract_tensor_cpu(file, "$(prefix).ssm_dt.bias")))
    
    # SSM norm
    ssm_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).ssm_norm.weight"))
    ssm_norm = ModelCPU.RMSNormCPU(ssm_norm_w, config.rms_norm_eps)
    
    # Dimensions - need to get from model
    # For Qwen3.5-0.8B: d_inner = 2048, num_v_heads = 16, num_k_heads = 16
    # head_k_dim = 128, head_v_dim = 128
    num_v_heads = length(ssm_a)
    num_k_heads = num_v_heads  # same for Qwen3.5
    d_inner = size(gate_proj, 1)
    head_v_dim = d_inner ÷ num_v_heads
    head_k_dim = size(ssm_alpha_weight, 1) ÷ num_k_heads  # Actually need to check this
    
    # conv_channels = d_inner + 2 * num_k_heads * head_k_dim
    conv_channels = size(in_proj, 1)
    conv_kernel = size(ssm_conv1d, 2)
    
    # Recompute head_k_dim from conv_channels
    # conv_channels = d_inner + 2 * num_k_heads * head_k_dim
    # head_k_dim = (conv_channels - d_inner) / (2 * num_k_heads)
    head_k_dim = (conv_channels - d_inner) ÷ (2 * num_k_heads)
    
    # State buffers
    conv_state = zeros(Float32, conv_channels, conv_kernel)
    h = zeros(Float32, head_v_dim, head_k_dim, num_v_heads)
    
    return ModelCPU.GatedDeltaNetCPU(
        layer_idx + 1,
        in_proj, gate_proj, ssm_out, ssm_conv1d,
        ssm_alpha_weight, ssm_beta_weight,
        ssm_a, ssm_dt_bias, ssm_norm,
        num_v_heads, num_k_heads, head_k_dim, head_v_dim, d_inner,
        conv_channels, conv_kernel,
        conv_state, h
    )
end

function load_attention_layer(file::GGUF.GGUFFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU)
    prefix = "blk.$(layer_idx)"
    
    # Load weights - transpose from GGUF format
    wq = Float32.(extract_tensor_cpu(file, "$(prefix).attn_q.weight"))'
    wk = Float32.(extract_tensor_cpu(file, "$(prefix).attn_k.weight"))'
    wv = Float32.(extract_tensor_cpu(file, "$(prefix).attn_v.weight"))'
    wo = Float32.(extract_tensor_cpu(file, "$(prefix).attn_output.weight"))'
    
    # Q/K norms
    q_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).attn_q_norm.weight"))
    q_norm = ModelCPU.RMSNormCPU(q_norm_w, config.rms_norm_eps)
    
    k_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).attn_k_norm.weight"))
    k_norm = ModelCPU.RMSNormCPU(k_norm_w, config.rms_norm_eps)
    
    # Get head dimensions from weights
    n_heads = size(wq, 1) ÷ config.head_dim
    n_kv = size(wk, 1) ÷ config.head_dim
    
    scale = 1.0f0 / sqrt(Float32(config.head_dim))
    
    return ModelCPU.FullAttentionCPU(
        layer_idx + 1,
        wq, wk, wv, wo,
        q_norm, k_norm,
        n_heads, n_kv, config.head_dim, scale
    )
end

function load_mlp(file::GGUF.GGUFFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU)
    prefix = "blk.$(layer_idx)"
    
    gate_weight = Float32.(extract_tensor_cpu(file, "$(prefix).ffn_gate.weight"))'
    up_weight = Float32.(extract_tensor_cpu(file, "$(prefix).ffn_up.weight"))'
    down_weight = Float32.(extract_tensor_cpu(file, "$(prefix).ffn_down.weight"))'
    
    return ModelCPU.MLPCPU(gate_weight, up_weight, down_weight)
end

end # module
