module Loader

using oneAPI
using ..GGUF
using ..Model

export load_weights

# --- Dequantization Helpers ---

const Q5_K_BLOCK_SIZE = 256

"""
Dequantize Q5_K. 
Block: 1 f16 delta + 1 f16 min + 32 bytes (low 4 bits) + 32 bytes (high 1 bit).
"""
function dequantize_q5_k(data::AbstractVector{UInt8}, num_elements::Int)
    block_bytes = 2 + 2 + 128 + 32 # 164 bytes
    num_blocks = num_elements ÷ Q5_K_BLOCK_SIZE
    out = Vector{Float32}(undef, num_elements)
    
    for b in 0:(num_blocks - 1)
        offset = b * block_bytes
        delta = Float32(reinterpret(Float16, @view data[offset+1:offset+2])[1])
        min_v = Float32(reinterpret(Float16, @view data[offset+3:offset+4])[1])
        
        low_bits = @view data[offset+5:offset+132]
        high_bits = @view data[offset+133:offset+164]
        
        for i in 0:255
            # Extract 4 low bits
            byte_idx = (i ÷ 2) + 1
            low = (i % 2 == 0) ? (low_bits[byte_idx] & 0x0F) : (low_bits[byte_idx] >> 4)
            
            # Extract 1 high bit
            high_byte_idx = (i ÷ 8) + 1
            high_bit_idx = (i % 8)
            high = (high_bits[high_byte_idx] >> high_bit_idx) & 0x01
            
            val = Int32(low) | (Int32(high) << 4)
            out[b * Q5_K_BLOCK_SIZE + i + 1] = Float32(val) * delta + min_v
        end
    end
    return out
end

const IQ2_XXS_BLOCK_SIZE = 256

"""
Dequantize IQ2_XXS (Simplified for Scaffold).
Uses the 16-bit scale and treats the rest as 2-bit signed indices.
"""
function dequantize_iq2_xxs(data::AbstractVector{UInt8}, num_elements::Int)
    block_bytes = 66 # 2 (f16 scale) + 64 (256 * 2 bits)
    num_blocks = num_elements ÷ IQ2_XXS_BLOCK_SIZE
    out = Vector{Float32}(undef, num_elements)
    
    for b in 0:(num_blocks - 1)
        offset = b * block_bytes
        scale = Float32(reinterpret(Float16, @view data[offset+1:offset+2])[1])
        
        # 64 bytes for 256 indices (2 bits each)
        indices = @view data[offset+3:offset+66]
        for i in 0:255
            byte_idx = (i ÷ 4) + 1
            shift = (i % 4) * 2
            val = (indices[byte_idx] >> shift) & 0x03
            # Simple mapping: 0 -> -1.5, 1 -> -0.5, 2 -> 0.5, 3 -> 1.5
            fval = (Float32(val) - 1.5f0)
            out[b * IQ2_XXS_BLOCK_SIZE + i + 1] = fval * scale
        end
    end
    return out
end

# --- Tensor Extraction ---

function extract_tensor(file::GGUF.GGUFFile, name::String)
    if !haskey(file.tensors, name)
        error("Tensor not found: $name")
    end
    info = file.tensors[name]
    num_elements = Int(prod(info.dimensions))
    start = Int(file.data_offset + info.offset) + 1
    
    data = if info.type == GGUF.GGML_TYPE_F32
        reinterpret(Float32, @view file.tensor_data[start:start+num_elements*4-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_F16
        Float32.(reinterpret(Float16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect)
    elseif info.type == GGUF.GGML_TYPE_Q5_K
        dequantize_q5_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_XXS
        dequantize_iq2_xxs(@view(file.tensor_data[start:end]), num_elements)
    else
        # Fallback for other quants: return zeros but don't crash
        @warn "Unsupported quant type $(info.type) for $name, using zeros"
        zeros(Float32, num_elements)
    end
    
    dims = Tuple(Int.(info.dimensions))
    # GGUF is typically [cols, rows], Julia is [rows, cols]
    # For linear weights, we want it as (out_features, in_features)
    # GGUF [in, out] -> Julia [in, out]
    # We might need to transpose depending on how we use it.
    # For now, just reshape.
    return oneArray(reshape(data, dims))
end

function load_weights(file::GGUF.GGUFFile, config::QwenConfig)
    println("  Loading embeddings...")
    embed = extract_tensor(file, "token_embd.weight")
    
    println("  Loading layers...")
    layers = DecoderLayer[]
    for i in 0:(config.num_hidden_layers - 1)
        prefix = "blk.$(i)"
        
        # Attention
        qkv_w = extract_tensor(file, "$(prefix).attn_qkv.weight")
        o_w   = extract_tensor(file, "$(prefix).attn_output.weight")
        
        q_b = haskey(file.tensors, "$(prefix).attn_qkv.bias") ? # Bias is rare in Qwen
              vec(extract_tensor(file, "$(prefix).attn_qkv.bias")) : nothing
              
        attn = Attention(qkv_w, o_w, q_b, nothing, nothing, 
                         config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        
        # Norms
        attn_norm_w = vec(extract_tensor(file, "$(prefix).attn_norm.weight"))
        attn_norm = RMSNorm(attn_norm_w, config.rms_norm_eps)
        
        # Qwen3.5 uses post_attention_norm instead of ffn_norm
        post_norm_key = haskey(file.tensors, "$(prefix).post_attention_norm.weight") ? 
                        "$(prefix).post_attention_norm.weight" : "$(prefix).ffn_norm.weight"
        post_norm_w = vec(extract_tensor(file, post_norm_key))
        post_norm = RMSNorm(post_norm_w, config.rms_norm_eps)
        
        # MLP
        gate = extract_tensor(file, "$(prefix).ffn_gate.weight")
        up   = extract_tensor(file, "$(prefix).ffn_up.weight")
        down = extract_tensor(file, "$(prefix).ffn_down.weight")
        mlp  = MLP(gate, up, down)
        
        # SSM (Optional)
        ssm = if haskey(file.tensors, "$(prefix).ssm_a")
            # Minimal SSM load for the scaffold
            a = extract_tensor(file, "$(prefix).ssm_a")
            # We skip others for now as the SSM block is a placeholder
            SSM(a, oneArray(zeros(Float32, 1, 1)), oneVector(zeros(Float32, 1)))
        else
            nothing
        end
        
        push!(layers, DecoderLayer(attn_norm, attn, ssm, post_norm, mlp))
    end
    
    println("  Loading final layers...")
    final_norm_w = vec(extract_tensor(file, "output_norm.weight"))
    final_norm = RMSNorm(final_norm_w, config.rms_norm_eps)
    
    lm_head = haskey(file.tensors, "output.weight") ? 
              extract_tensor(file, "output.weight") : embed
              
    rope = RotaryEmbedding(config.head_dim; base=config.rope_theta)
    
    return QwenModel(config, embed, layers, final_norm, lm_head, rope)
end

end # module
