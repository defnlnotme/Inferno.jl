module Loader

using oneAPI
using ..GGUF
using ..Model

export load_weights

using ..Dequant

function extract_tensor(file::GGUF.GGUFFile, name::String)
    if !haskey(file.tensors, name)
        @warn "Tensor $name not found, using zero surrogate"
        return zeros(Float16, 1, 1) |> oneArray # Tiny to avoid crash, will pad later
    end
    info = file.tensors[name]
    num_elements = Int(prod(info.dimensions))
    start = Int(file.data_offset + info.offset) + 1

    data32 = if info.type == GGUF.GGML_TYPE_F32
        reinterpret(Float32, @view file.tensor_data[start:start+num_elements*4-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_F16
        Float32.(reinterpret(Float16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect)
    elseif info.type == GGUF.GGML_TYPE_Q5_K
        dequantize_q5_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q4_K
        dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q8_0
        dequantize_q8_0(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_XXS
        dims = Tuple(Int.(info.dimensions))
        inner = dims[1]
        outer = length(dims) > 1 ? dims[2] : 1
        # Optimization: Upload IQ2_XXS weights to GPU once during loading
        # to avoid massive memory churn during inference hot-path.
        raw_data = @view file.tensor_data[start:start + num_elements * 66 ÷ 256 - 1]
        gpu_data = oneArray(collect(raw_data))
        return Model.IQ2XXSMatrix(gpu_data, inner, outer)
    elseif info.type == GGUF.GGML_TYPE_IQ2_XS
        dequantize_iq2_xs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_S
        dequantize_iq2_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_XXS
        dequantize_iq3_xxs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_S
        dequantize_iq3_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q2_K
        dequantize_q2_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q3_K
        dequantize_q3_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q6_K
        dequantize_q6_k(@view(file.tensor_data[start:end]), num_elements)
    else
        @warn "Unhandled type $(info.type). Zeroing."
        zeros(Float32, num_elements)
    end

    dims = Tuple(Int.(info.dimensions))
    inner = dims[1]
    outer = length(dims) > 1 ? dims[2] : 1

    # Return as CPU Float16 matrix for stability during loading
    return reshape(Float16.(data32), inner, outer)
end

function get_weight(file::GGUF.GGUFFile, name::String)
    tensor = extract_tensor(file, name)
    if tensor isa Model.IQ2XXSMatrix
        return tensor
    else
        return oneArray(Float32.(tensor'))
    end
end

function get_bias_or_norm(file::GGUF.GGUFFile, name::String)
    tensor = extract_tensor(file, name)
    return vec(collect(Float32.(tensor)))
end

function load_weights(file::GGUF.GGUFFile, config::Model.QwenConfig)

    embed = extract_tensor(file, "token_embd.weight")


    layers = Model.DecoderLayer[]
    for i in 0:(config.num_hidden_layers-1)
        prefix = "blk.$(i)"
        is_ssm = haskey(file.tensors, "$(prefix).ssm_a")

        in_norm_w = get_bias_or_norm(file, "$(prefix).attn_norm.weight")
        in_norm = Model.RMSNorm(oneArray(in_norm_w), config.rms_norm_eps)

        post_norm_key = haskey(file.tensors, "$(prefix).post_attention_norm.weight") ?
                        "$(prefix).post_attention_norm.weight" : "$(prefix).attn_norm.weight"
        post_norm_w = get_bias_or_norm(file, post_norm_key)
        post_norm = Model.RMSNorm(oneArray(post_norm_w), config.rms_norm_eps)

        op = if is_ssm
            in_proj   = get_weight(file, "$(prefix).attn_qkv.weight")
            gate_proj = get_weight(file, "$(prefix).attn_gate.weight")
            ssm_out   = get_weight(file, "$(prefix).ssm_out.weight")

            ssm_a      = oneArray(get_bias_or_norm(file, "$(prefix).ssm_a"))
            ssm_alpha  = get_weight(file, "$(prefix).ssm_alpha.weight")
            ssm_beta   = get_weight(file, "$(prefix).ssm_beta.weight")
            ssm_conv1d_raw = extract_tensor(file, "$(prefix).ssm_conv1d.weight")
            ssm_conv1d_f32 = oneArray(collect(Float32.(ssm_conv1d_raw)))
            oneAPI.synchronize()
            ssm_dt_bias    = oneArray(get_bias_or_norm(file, "$(prefix).ssm_dt.bias"))
            ssm_norm_w     = get_bias_or_norm(file, "$(prefix).ssm_norm.weight")
            ssm_norm       = Model.RMSNorm(oneArray(ssm_norm_w), config.rms_norm_eps)

            # conv_channels = d_inner + 2 * num_k_heads * head_k_dim
            conv_kernel = 4  # ssm.conv_kernel
            num_v_heads = config.ssm_time_step_rank  # 16
            num_k_heads = config.ssm_group_count      # 16
            head_k_dim  = config.ssm_state_size       # 128
            head_v_dim  = config.ssm_inner_size ÷ num_v_heads  # 128
            conv_channels = config.ssm_inner_size + 2 * num_k_heads * head_k_dim  # 6144

            # Keep SSM state on CPU for efficiency - avoid GPU-CPU transfers
            conv_state = zeros(Float32, conv_kernel, conv_channels)  # CPU ring buffer
            ssm_state  = zeros(Float32, head_v_dim, head_k_dim, num_v_heads)  # CPU state matrix

            ssm_conv1d_cpu = collect(Float32.(ssm_conv1d_raw))

            Model.GatedDeltaNet(in_proj, gate_proj, ssm_out,
                ssm_a, ssm_alpha, ssm_beta, ssm_conv1d_f32, ssm_conv1d_cpu, ssm_dt_bias, ssm_norm,
                conv_state, ssm_state,
                num_v_heads, num_k_heads, head_k_dim, head_v_dim, config.ssm_inner_size)

        else
            qw = get_weight(file, "$(prefix).attn_q.weight")
            kw = get_weight(file, "$(prefix).attn_k.weight")
            vw = get_weight(file, "$(prefix).attn_v.weight")
            ow = get_weight(file, "$(prefix).attn_output.weight")

            q_norm_w = get_bias_or_norm(file, "$(prefix).attn_q_norm.weight")
            k_norm_w = get_bias_or_norm(file, "$(prefix).attn_k_norm.weight")
            q_norm = Model.RMSNorm(oneArray(q_norm_w), config.rms_norm_eps)
            k_norm = Model.RMSNorm(oneArray(k_norm_w), config.rms_norm_eps)

            # Q output has packed Q+gate: size = head_dim * 2 * n_heads
            # Use size because get_weight returns IQ2XXSMatrix or Float32 matrix
            Model.FullAttention(qw, kw, vw, ow, q_norm, k_norm, config)
        end

        gate_w = get_weight(file, "$(prefix).ffn_gate.weight")
        up_w   = get_weight(file, "$(prefix).ffn_up.weight")
        down_w = get_weight(file, "$(prefix).ffn_down.weight")
        mlp    = Model.MLP(gate_w, up_w, down_w)

        push!(layers, Model.DecoderLayer(in_norm, op, post_norm, mlp, is_ssm))
    end

    final_norm_w = get_bias_or_norm(file, "output_norm.weight")
    final_norm = Model.RMSNorm(oneArray(final_norm_w), config.rms_norm_eps)

    lm_head_raw = haskey(file.tensors, "output.weight") ?
                  extract_tensor(file, "output.weight") : embed
    lm_head = Float32.(lm_head_raw)  # Keep on CPU for now

    rope = Model.RotaryEmbedding(config.head_dim; base=config.rope_theta)

    return Model.QwenModel(config, Float32.(embed), layers, final_norm, lm_head, rope)
end

end # module
