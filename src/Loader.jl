module Loader

using oneAPI
using ..GGUF
using ..Model

export load_weights

using ..Dequant

function extract_tensor(file::GGUF.GGUFFile, name::String; imatrix::Union{GGUF.GGUFFile, Nothing}=nothing)
    if !haskey(file.tensors, name)
        @warn "Tensor $name not found, using zero surrogate"
        return zeros(Float16, 1, 1) |> oneArray # Tiny to avoid crash, will pad later
    end
    
    if imatrix !== nothing && haskey(imatrix.tensors, name)
        # In a real GGUF loader like llama.cpp, imatrix data would be used 
        # to guide the importance-aware quantization. 
        # Here we just acknowledge its presence for the tensor.
        # println("  Applying imatrix for $name")
    end
    info = file.tensors[name]
    num_elements = Int(prod(info.dimensions))
    start = Int(file.data_offset + info.offset) + 1

    data32 = if info.type == GGUF.GGML_TYPE_F32
        reinterpret(Float32, @view file.tensor_data[start:start+num_elements*4-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_F16
        Float32.(reinterpret(Float16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect)
    elseif info.type == GGUF.GGML_TYPE_BF16
        # BF16 to Float32 conversion
        raw = reinterpret(UInt16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect
        Float32.(reinterpret(Float32, [UInt32(r) << 16 for r in raw]))
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
    elseif info.type == GGUF.GGML_TYPE_IQ4_XS
        dequantize_iq4_xs(@view(file.tensor_data[start:end]), num_elements)
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

function get_weight(file::GGUF.GGUFFile, name::String; imatrix::Union{GGUF.GGUFFile, Nothing}=nothing)
    tensor = extract_tensor(file, name; imatrix=imatrix)
    if tensor isa Model.IQ2XXSMatrix
        return tensor
    else
        return oneArray(Float32.(tensor'))
    end
end

function get_bias_or_norm(file::GGUF.GGUFFile, name::String; imatrix::Union{GGUF.GGUFFile, Nothing}=nothing)
    tensor = extract_tensor(file, name; imatrix=imatrix)
    return vec(collect(Float32.(tensor)))
end

function load_weights(file::GGUF.GGUFFile, config::Model.QwenConfig; 
                      imatrix::Union{GGUF.GGUFFile, Nothing}=nothing,
                      mmproj::Union{GGUF.GGUFFile, Nothing}=nothing)
    arch = config.architecture
    embed = extract_tensor(file, "token_embd.weight"; imatrix=imatrix)

    layers = Model.DecoderLayer[]
    for i in 0:(config.num_hidden_layers-1)
        print(".")
        flush(stdout)
        prefix = "blk.$(i)"
        is_ssm = haskey(file.tensors, "$(prefix).ssm_a")

        # Standardize normalization key search
        in_norm_key = haskey(file.tensors, "$(prefix).attn_norm.weight") ? "$(prefix).attn_norm.weight" :
                      haskey(file.tensors, "$(prefix).input_layernorm.weight") ? "$(prefix).input_layernorm.weight" :
                      "$(prefix).attn_norm.weight"

        in_norm_w = get_bias_or_norm(file, in_norm_key; imatrix=imatrix)
        in_norm = Model.RMSNorm(oneArray(in_norm_w), config.rms_norm_eps)

        post_norm_key = haskey(file.tensors, "$(prefix).post_attention_norm.weight") ? "$(prefix).post_attention_norm.weight" :
                        haskey(file.tensors, "$(prefix).ffn_norm.weight") ? "$(prefix).ffn_norm.weight" :
                        haskey(file.tensors, "$(prefix).post_attention_layernorm.weight") ? "$(prefix).post_attention_layernorm.weight" :
                        "$(prefix).attn_norm.weight"
        post_norm_w = get_bias_or_norm(file, post_norm_key; imatrix=imatrix)
        post_norm = Model.RMSNorm(oneArray(post_norm_w), config.rms_norm_eps)

        op = if is_ssm
            in_proj   = get_weight(file, "$(prefix).attn_qkv.weight"; imatrix=imatrix)
            gate_proj = get_weight(file, "$(prefix).attn_gate.weight"; imatrix=imatrix)
            ssm_out   = get_weight(file, "$(prefix).ssm_out.weight"; imatrix=imatrix)

            ssm_a      = oneArray(get_bias_or_norm(file, "$(prefix).ssm_a"; imatrix=imatrix))
            ssm_alpha  = get_weight(file, "$(prefix).ssm_alpha.weight"; imatrix=imatrix)
            ssm_beta   = get_weight(file, "$(prefix).ssm_beta.weight"; imatrix=imatrix)
            ssm_conv1d_raw = extract_tensor(file, "$(prefix).ssm_conv1d.weight"; imatrix=imatrix)
            ssm_conv1d_f32 = oneArray(collect(Float32.(ssm_conv1d_raw)))
            oneAPI.synchronize()
            ssm_dt_bias    = oneArray(get_bias_or_norm(file, "$(prefix).ssm_dt.bias"; imatrix=imatrix))
            ssm_norm_w     = get_bias_or_norm(file, "$(prefix).ssm_norm.weight"; imatrix=imatrix)
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
        elseif arch == :deepseek2
            # Simplified MLA loading
            q_a_proj = get_weight(file, "$(prefix).attn_q_a.weight")
            q_a_norm = Model.RMSNorm(oneArray(get_bias_or_norm(file, "$(prefix).attn_q_a_norm.weight")), config.rms_norm_eps)
            q_b_proj = get_weight(file, "$(prefix).attn_q_b.weight")
            kv_a_proj_with_mqa = get_weight(file, "$(prefix).attn_kv_a_mqa.weight")
            kv_a_norm = Model.RMSNorm(oneArray(get_bias_or_norm(file, "$(prefix).attn_kv_a_norm.weight")), config.rms_norm_eps)
            kv_b_proj = get_weight(file, "$(prefix).attn_kv_b.weight")
            wo = get_weight(file, "$(prefix).attn_output.weight")

            Model.MLAttention(q_a_proj, q_a_norm, q_b_proj, kv_a_proj_with_mqa, kv_a_norm, kv_b_proj, wo,
                config.num_attention_heads, config.head_dim, config.q_lora_rank, config.kv_lora_rank,
                config.qk_rope_head_dim, config.v_head_dim, 1.0f0/sqrt(Float32(config.head_dim)))
        else
            wqkv = arch == :phi3 ? get_weight(file, "$(prefix).attn_qkv.weight"; imatrix=imatrix) : nothing
            qw = arch != :phi3 ? get_weight(file, "$(prefix).attn_q.weight"; imatrix=imatrix) : nothing
            kw = arch != :phi3 ? get_weight(file, "$(prefix).attn_k.weight"; imatrix=imatrix) : nothing
            vw = arch != :phi3 ? get_weight(file, "$(prefix).attn_v.weight"; imatrix=imatrix) : nothing
            ow = get_weight(file, "$(prefix).attn_output.weight"; imatrix=imatrix)

            q_norm_w = haskey(file.tensors, "$(prefix).attn_q_norm.weight") ? get_bias_or_norm(file, "$(prefix).attn_q_norm.weight"; imatrix=imatrix) : nothing
            k_norm_w = haskey(file.tensors, "$(prefix).attn_k_norm.weight") ? get_bias_or_norm(file, "$(prefix).attn_k_norm.weight"; imatrix=imatrix) : nothing
            q_norm = isnothing(q_norm_w) ? nothing : Model.RMSNorm(oneArray(q_norm_w), config.rms_norm_eps)
            k_norm = isnothing(k_norm_w) ? nothing : Model.RMSNorm(oneArray(k_norm_w), config.rms_norm_eps)

            n_heads = config.num_attention_heads
            n_kv    = config.num_key_value_heads

            Model.FullAttention(arch, qw, kw, vw, wqkv, ow, q_norm, k_norm, n_heads, n_kv, config.head_dim)
        end

        # MLP or MoE
        mlp = if haskey(file.tensors, "$(prefix).ffn_gate_exps.0.weight") || haskey(file.tensors, "$(prefix).ffn_gate_exps.weight")
            # MoE
            gate = get_weight(file, "$(prefix).ffn_gate.weight"; imatrix=imatrix)
            n_exp = config.num_experts
            experts_gate = [get_weight(file, "$(prefix).ffn_gate_exps.$(j-1).weight"; imatrix=imatrix) for j in 1:n_exp]
            experts_up = [get_weight(file, "$(prefix).ffn_up_exps.$(j-1).weight"; imatrix=imatrix) for j in 1:n_exp]
            experts_down = [get_weight(file, "$(prefix).ffn_down_exps.$(j-1).weight"; imatrix=imatrix) for j in 1:n_exp]
            Model.MoE(gate, experts_gate, experts_up, experts_down, n_exp, config.num_experts_per_tok)
        else
            gate_w = get_weight(file, "$(prefix).ffn_gate.weight"; imatrix=imatrix)
            up_w   = get_weight(file, "$(prefix).ffn_up.weight"; imatrix=imatrix)
            down_w = get_weight(file, "$(prefix).ffn_down.weight"; imatrix=imatrix)
            Model.MLP(gate_w, up_w, down_w)
        end

        push!(layers, Model.DecoderLayer(in_norm, op, post_norm, mlp, is_ssm))
    end
    println()

    final_norm_w = get_bias_or_norm(file, "output_norm.weight"; imatrix=imatrix)
    final_norm = Model.RMSNorm(oneArray(final_norm_w), config.rms_norm_eps)

    lm_head_raw = haskey(file.tensors, "output.weight") ?
                  extract_tensor(file, "output.weight"; imatrix=imatrix) : embed
    lm_head = Float32.(lm_head_raw)  # Keep on CPU for now

    rope_dim = Int(get(file.metadata, "$(arch).rope.dimension_count", config.head_dim))
    rope = Model.RotaryEmbedding(rope_dim; base=config.rope_theta)

    mmproj_data = nothing
    if mmproj !== nothing
        println("Integrating mmproj tensors...")
        mmproj_data = Dict{String, Any}()
        for name in keys(mmproj.tensors)
            mmproj_data[name] = get_weight(mmproj, name)
        end
    end

    return Model.QwenModel(config, Float32.(embed), layers, final_norm, lm_head, rope, mmproj_data)
end

end # module
