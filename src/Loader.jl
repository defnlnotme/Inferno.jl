module Loader

using oneAPI
using ..GGUF
using ..Model

export load_weights

using Libdl

const GGML_LIB_PATH = "/var/home/fra/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-vulkan-avx2-1.104.2/libggml-cpu.so"
const GGML_HND_REF = Ref{Ptr{Cvoid}}(C_NULL)

function get_ggml_handle()
    if GGML_HND_REF[] == C_NULL
        try
            hnd = dlopen(GGML_LIB_PATH)
            if hnd != C_NULL
                init_sym = dlsym(hnd, :ggml_quantize_init; throw_error=false)
                if init_sym != C_NULL
                    ccall(init_sym, Cvoid, ())
                end
                GGML_HND_REF[] = hnd
            end
        catch e
            @warn "Failed to load GGML library" e
        end
    end
    return GGML_HND_REF[]
end

function ggml_dequantize(sym_name::Symbol, data::AbstractVector{UInt8}, num_elements::Int)
    hnd = get_ggml_handle()
    if hnd == C_NULL
        @warn "GGML native library not found. Falling back to zeros."
        return zeros(Float32, num_elements)
    end
    
    sym = dlsym(hnd, sym_name; throw_error=false)
    if sym == C_NULL
        @warn "Symbol $sym_name not found in GGML library. Falling back to zeros."
        return zeros(Float32, num_elements)
    end
    
    out = Vector{Float32}(undef, num_elements)
    GC.@preserve data out begin
        ptr_in = pointer(data)
        ptr_out = pointer(out)
        ccall(sym, Cvoid, (Ptr{UInt8}, Ptr{Float32}, Int64), ptr_in, ptr_out, num_elements)
    end
    return out
end

# Dequantization wrappers targeting the native lib functions
dequantize_q5_k(data, n) = ggml_dequantize(:dequantize_row_q5_K, data, n)
dequantize_q4_k(data, n) = ggml_dequantize(:dequantize_row_q4_K, data, n)
dequantize_iq2_xxs(data, n) = ggml_dequantize(:dequantize_row_iq2_xxs, data, n)
dequantize_iq3_xxs(data, n) = ggml_dequantize(:dequantize_row_iq3_xxs, data, n)
dequantize_iq3_s(data, n) = ggml_dequantize(:dequantize_row_iq3_s, data, n)
dequantize_iq2_s(data, n) = ggml_dequantize(:dequantize_row_iq2_s, data, n)
dequantize_q2_k(data, n) = ggml_dequantize(:dequantize_row_q2_K, data, n)
dequantize_q3_k(data, n) = ggml_dequantize(:dequantize_row_q3_K, data, n)
dequantize_q8_0(data, n) = ggml_dequantize(:dequantize_row_q8_0, data, n)

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
    elseif info.type == GGUF.GGML_TYPE_Q3_K
        dequantize_q3_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q2_K
        dequantize_q2_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q8_0
        dequantize_q8_0(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_XXS
        dequantize_iq2_xxs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_S
        dequantize_iq2_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_XXS
        dequantize_iq3_xxs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_S
        dequantize_iq3_s(@view(file.tensor_data[start:end]), num_elements)
    else
        @warn "Unhandled type $(info.type). Zeroing."
        zeros(Float32, num_elements)
    end

    dims = Tuple(Int.(info.dimensions))
    cpu_data = reshape(data32, dims)

    # More stable allocation/copy
    gpu_data = oneArray{Float16}(undef, size(cpu_data))
    copyto!(gpu_data, Float16.(cpu_data))

    oneAPI.synchronize()
    return gpu_data
end

function load_weights(file::GGUF.GGUFFile, config::Model.QwenConfig)
    println("  Loading weights (targeting Float16)...")

    embed = extract_tensor(file, "token_embd.weight")

    layers = Model.DecoderLayer[]
    for i in 0:(config.num_hidden_layers-1)
        prefix = "blk.$(i)"
        is_ssm = haskey(file.tensors, "$(prefix).ssm_a")
        print("  Layer $i ($(is_ssm ? "SSM" : "Attn"))... ")

        in_norm_w = vec(collect(Float32.(extract_tensor(file, "$(prefix).attn_norm.weight"))))
        in_norm = Model.RMSNorm(oneArray(Float16.(in_norm_w)), config.rms_norm_eps)

        post_norm_key = haskey(file.tensors, "$(prefix).post_attention_norm.weight") ?
                        "$(prefix).post_attention_norm.weight" : "$(prefix).attn_norm.weight"
        post_norm_w = vec(collect(Float32.(extract_tensor(file, post_norm_key))))
        post_norm = Model.RMSNorm(oneArray(Float16.(post_norm_w)), config.rms_norm_eps)

        op = if is_ssm
            in_proj   = extract_tensor(file, "$(prefix).attn_qkv.weight")
            gate_proj = extract_tensor(file, "$(prefix).attn_gate.weight")
            ssm_out   = extract_tensor(file, "$(prefix).ssm_out.weight")

            ssm_a_cpu      = oneArray(vec(collect(Float32.(extract_tensor(file, "$(prefix).ssm_a")))))
            ssm_alpha      = extract_tensor(file, "$(prefix).ssm_alpha.weight")
            ssm_beta       = extract_tensor(file, "$(prefix).ssm_beta.weight")
            ssm_conv1d_raw = extract_tensor(file, "$(prefix).ssm_conv1d.weight")
            ssm_conv1d_f32 = oneArray(collect(Float32.(ssm_conv1d_raw)))
            ssm_dt_bias    = oneArray(vec(collect(Float32.(extract_tensor(file, "$(prefix).ssm_dt.bias")))))
            ssm_norm_w     = vec(collect(Float32.(extract_tensor(file, "$(prefix).ssm_norm.weight"))))
            ssm_norm       = Model.RMSNorm(oneArray(Float16.(ssm_norm_w)), config.rms_norm_eps)

            # conv_channels = d_inner + 2 * num_k_heads * head_k_dim
            conv_kernel = 4  # ssm.conv_kernel
            num_v_heads = config.ssm_time_step_rank  # 16
            num_k_heads = config.ssm_group_count      # 16
            head_k_dim  = config.ssm_state_size       # 128
            head_v_dim  = config.ssm_inner_size ÷ num_v_heads  # 128
            conv_channels = config.ssm_inner_size + 2 * num_k_heads * head_k_dim  # 6144

            conv_state = zeros(Float32, conv_kernel, conv_channels)  # CPU ring buffer
            ssm_state  = zeros(Float32, head_v_dim, head_k_dim, num_v_heads)  # CPU state matrix

            Model.GatedDeltaNet(in_proj, gate_proj, ssm_out,
                ssm_a_cpu, ssm_alpha, ssm_beta, ssm_conv1d_f32, ssm_dt_bias, ssm_norm,
                conv_state, ssm_state,
                num_v_heads, num_k_heads, head_k_dim, head_v_dim, config.ssm_inner_size)
        else
            qw = extract_tensor(file, "$(prefix).attn_q.weight")
            kw = extract_tensor(file, "$(prefix).attn_k.weight")
            vw = extract_tensor(file, "$(prefix).attn_v.weight")
            ow = extract_tensor(file, "$(prefix).attn_output.weight")

            q_norm_w = vec(collect(Float32.(extract_tensor(file, "$(prefix).attn_q_norm.weight"))))
            k_norm_w = vec(collect(Float32.(extract_tensor(file, "$(prefix).attn_k_norm.weight"))))
            q_norm = Model.RMSNorm(oneArray(Float16.(q_norm_w)), config.rms_norm_eps)
            k_norm = Model.RMSNorm(oneArray(Float16.(k_norm_w)), config.rms_norm_eps)

            # Q output has packed Q+gate: size = head_dim * 2 * n_heads
            n_heads = size(qw, 2) ÷ (config.head_dim * 2)
            n_kv    = size(kw, 2) ÷ config.head_dim

            Model.FullAttention(qw, kw, vw, ow, q_norm, k_norm, n_heads, n_kv, config.head_dim)
        end

        gate_w = extract_tensor(file, "$(prefix).ffn_gate.weight")
        up_w   = extract_tensor(file, "$(prefix).ffn_up.weight")
        down_w = extract_tensor(file, "$(prefix).ffn_down.weight")
        mlp    = Model.MLP(gate_w, up_w, down_w)

        push!(layers, Model.DecoderLayer(in_norm, op, post_norm, mlp, is_ssm))
        println("ok")

        oneAPI.synchronize()
        GC.gc()
    end

    final_norm_w = vec(collect(Float32.(extract_tensor(file, "output_norm.weight"))))
    final_norm = Model.RMSNorm(oneArray(Float16.(final_norm_w)), config.rms_norm_eps)

    lm_head = haskey(file.tensors, "output.weight") ?
              extract_tensor(file, "output.weight") : embed

    rope = Model.RotaryEmbedding(config.head_dim; base=config.rope_theta)

    oneAPI.synchronize()
    GC.gc()

    return Model.QwenModel(config, embed, layers, final_norm, lm_head, rope)
end

end # module
