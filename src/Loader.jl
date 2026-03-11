module Loader

using oneAPI
using ..GGUF
using ..Model

export load_weights

# --- Dequantization Helpers (Float32 on CPU, then cast to Float16 for GPU) ---

function dequantize_q5_k(data::AbstractVector{UInt8}, num_elements::Int)
    block_bytes = 176
    num_blocks = num_elements ÷ 256
    out = Vector{Float32}(undef, num_elements)
    for b in 0:(num_blocks-1)
        offset = b * block_bytes
        d = Float32(reinterpret(Float16, @view data[offset+1:offset+2])[1])
        dmin = Float32(reinterpret(Float16, @view data[offset+3:offset+4])[1])
        qh = @view data[offset+17:offset+48]
        qs = @view data[offset+49:offset+176]
        for i in 0:255
            low = (i < 128) ? (qs[i+1] & 0x0F) : (qs[i-127] >> 4)
            high = (qh[(i÷8)+1] >> (i % 8)) & 0x01
            val = Int32(low) | (Int32(high) << 4)
            out[b*256+i+1] = Float32(val) * d + dmin
        end
    end
    return out
end

function dequantize_q4_k(data::AbstractVector{UInt8}, num_elements::Int)
    block_bytes = 176
    num_blocks = num_elements ÷ 256
    out = Vector{Float32}(undef, num_elements)
    for b in 0:(num_blocks-1)
        offset = b * block_bytes
        d = Float32(reinterpret(Float16, @view data[offset+1:offset+2])[1])
        qs = @view data[offset+49:offset+176]
        for i in 0:255
            val = (i < 128) ? (qs[i+1] & 0x0F) : (qs[i-127] >> 4)
            out[b*256+i+1] = Float32(val) * d
        end
    end
    return out
end

# Fallback for complex i-quants: just return zeros for now but warn
function dequantize_iq_fallback(num_elements::Int, type_name::String)
    @warn "Using zero fallback for $type_name"
    return zeros(Float32, num_elements)
end

const IQ2_XXS_BLOCK_SIZE = 256
function dequantize_iq2_xxs(data::AbstractVector{UInt8}, num_elements::Int)
    block_bytes = 66
    num_blocks = num_elements ÷ IQ2_XXS_BLOCK_SIZE
    out = Vector{Float32}(undef, num_elements)
    for b in 0:(num_blocks-1)
        offset = b * block_bytes
        d = Float32(reinterpret(Float16, @view data[offset+1:offset+2])[1])
        qs = @view data[offset+3:offset+66]
        for i in 0:255
            byte_idx = (i ÷ 4) + 1
            bit_idx = (i % 4) * 2
            val = (qs[byte_idx] >> bit_idx) & 0x03
            out[b*IQ2_XXS_BLOCK_SIZE+i+1] = (Float32(val) - 1.5f0) * d
        end
    end
    return out
end

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
    elseif info.type == GGUF.GGML_TYPE_IQ2_XXS
        dequantize_iq2_xxs(@view(file.tensor_data[start:end]), num_elements)
    else
        dequantize_iq_fallback(num_elements, string(info.type))
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

    # Load all tensors in a loop to manage memory
    embed = extract_tensor(file, "token_embd.weight")

    layers = Model.DecoderLayer[]
    for i in 0:(config.num_hidden_layers-1)
        prefix = "blk.$(i)"

        in_norm_w = vec(extract_tensor(file, "$(prefix).attn_norm.weight"))
        in_norm = Model.RMSNorm(in_norm_w, config.rms_norm_eps)

        post_norm_key = haskey(file.tensors, "$(prefix).post_attention_norm.weight") ?
                        "$(prefix).post_attention_norm.weight" : "$(prefix).attn_norm.weight"
        post_norm_w = vec(extract_tensor(file, post_norm_key))
        post_norm = Model.RMSNorm(post_norm_w, config.rms_norm_eps)

        op = if any(k -> contains(k, "$(prefix).attn_qkv"), keys(file.tensors))
            qkv = extract_tensor(file, "$(prefix).attn_qkv.weight")
            gate = extract_tensor(file, "$(prefix).attn_gate.weight")
            out = extract_tensor(file, "$(prefix).ssm_out.weight")
            Model.HybridBlock(qkv, gate, out)
        else
            qw = extract_tensor(file, "$(prefix).attn_q.weight")
            kw = extract_tensor(file, "$(prefix).attn_k.weight")
            vw = extract_tensor(file, "$(prefix).attn_v.weight")
            ow = extract_tensor(file, "$(prefix).attn_output.weight")
            Model.Attention(qw, kw, vw, ow, config.num_attention_heads, config.num_key_value_heads)
        end

        gate = extract_tensor(file, "$(prefix).ffn_gate.weight")
        up = extract_tensor(file, "$(prefix).ffn_up.weight")
        down = extract_tensor(file, "$(prefix).ffn_down.weight")
        mlp = Model.MLP(gate, up, down)

        push!(layers, Model.DecoderLayer(in_norm, op, post_norm, mlp))

        if i % 2 == 0
            oneAPI.synchronize()
            GC.gc()
        end
    end

    final_norm_w = vec(extract_tensor(file, "output_norm.weight"))
    final_norm = Model.RMSNorm(final_norm_w, config.rms_norm_eps)

    lm_head = haskey(file.tensors, "output.weight") ?
              extract_tensor(file, "output.weight") : embed

    rope = Model.RotaryEmbedding(config.head_dim; base=config.rope_theta)

    oneAPI.synchronize()
    GC.gc()

    return Model.QwenModel(config, embed, layers, final_norm, lm_head, rope)
end

end # module
