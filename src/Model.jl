module Model

using oneAPI
using LinearAlgebra

export QwenConfig, QwenModel, KVCache, forward!, RMSNorm, MLP, Attention, DecoderLayer

# --- Configuration ---
Base.@kwdef struct QwenConfig
    vocab_size::Int = 151936
    hidden_size::Int = 4096
    intermediate_size::Int = 11008
    num_hidden_layers::Int = 32
    num_attention_heads::Int = 32
    num_key_value_heads::Int = 32
    head_dim::Int = 128
    rms_norm_eps::Float32 = 1e-6
    rope_theta::Float32 = 1000000.0
    max_position_embeddings::Int = 32768
end

# --- Normalization ---
struct RMSNorm
    weight::oneVector{Float32}
    eps::Float32
end

function (norm::RMSNorm)(x::oneMatrix{Float32})
    # x is (hidden, seq)
    # RMS = sqrt(mean(x^2) + eps)
    rms = sqrt.(vec(sum(x.^2, dims=1)) ./ size(x, 1) .+ norm.eps)
    return (x ./ rms') .* norm.weight
end

# --- Rotary Embedding (RoPE) ---
struct RotaryEmbedding
    dim::Int
    base::Float32
    inv_freq::Vector{Float32}
end

function RotaryEmbedding(dim::Int; base=1000000.0)
    inv_freq = 1.0f0 ./ (base .^ (range(0, stop=dim-1, step=2) ./ dim))
    return RotaryEmbedding(dim, Float32(base), inv_freq)
end

function (rope::RotaryEmbedding)(x::oneArray{Float32, 3}, pos::Int)
    # x is (head_dim, n_heads, seq)
    dim, n_heads, seq = size(x)
    
    # In a real engine, we use precomputed cos/sin caches
    # For now, compute on the fly (slow but functional for scaffold)
    # pos is the starting position in the sequence
    out = copy(x)
    for s in 1:seq
        curr_pos = Float32(pos + s - 1)
        for i in 1:(dim÷2)
            theta = curr_pos * rope.inv_freq[i]
            cos_t = cos(theta)
            sin_t = sin(theta)
            
            for h in 1:n_heads
                v1 = x[2i-1, h, s]
                v2 = x[2i, h, s]
                out[2i-1, h, s] = v1 * cos_t - v2 * sin_t
                out[2i, h, s]   = v1 * sin_t + v2 * cos_t
            end
        end
    end
    return out
end

# --- KV Cache ---
mutable struct KVCache
    k::oneArray{Float32, 3} # (head_dim, n_kv_heads, max_seq)
    v::oneArray{Float32, 3}
    max_seq::Int
end

function init_kv_cache(head_dim, n_kv_heads, max_seq)
    k = zeros(Float32, head_dim, n_kv_heads, max_seq) |> oneArray
    v = zeros(Float32, head_dim, n_kv_heads, max_seq) |> oneArray
    return KVCache(k, v, max_seq)
end

function update!(cache::KVCache, k::oneArray{Float32, 3}, v::oneArray{Float32, 3}, pos::Int)
    seq = size(k, 3)
    @views cache.k[:, :, pos:pos+seq-1] .= k
    @views cache.v[:, :, pos:pos+seq-1] .= v
end

function get_full(cache::KVCache, current_len::Int)
    @views return cache.k[:, :, 1:current_len], cache.v[:, :, 1:current_len]
end

# --- MLP ---
struct MLP
    gate::oneArray{Float32, 2}
    up::oneArray{Float32, 2}
    down::oneArray{Float32, 2}
end

function (mlp::MLP)(x::oneMatrix{Float32})
    # SwiGLU: (swish(x*gate) * (x*up)) * down
    # x is (hidden, seq)
    g = (x' * mlp.gate)'
    u = (x' * mlp.up)'
    
    # Swish
    g .= g .* (1.0f0 ./ (1.0f0 .+ exp.(-g)))
    
    return ((g .* u)' * mlp.down)'
end

# --- SSM Block (Simplified Mamba-2) ---
struct SSM
    # Significant reduction for scaffold: only major weights
    a_weight::oneArray{Float32, 2}
    conv1d::oneArray{Float32, 2}
    dt_bias::oneVector{Float32}
    # Optional states would go here
end

function (ssm::SSM)(x::oneMatrix{Float32})
    # Placeholder: Identity for scaffold
    # Real Mamba-2 involves convolution, SSM projection, etc.
    return x
end

# --- Attention ---
struct Attention
    qkv_weight::oneArray{Float32, 2}
    o_weight::oneArray{Float32, 2}
    q_bias::Union{Nothing, oneVector{Float32}}
    k_bias::Union{Nothing, oneVector{Float32}}
    v_bias::Union{Nothing, oneVector{Float32}}
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
end

function (attn::Attention)(x::oneMatrix{Float32}, pos::Int, rope::RotaryEmbedding, kv_cache::KVCache)
    h, seq = size(x)
    
    # Project QKV combined
    qkv = (x' * attn.qkv_weight)' # (qkv_hidden, seq)
    
    q_size = attn.n_heads * attn.head_dim
    kv_size = attn.n_kv_heads * attn.head_dim
    
    q = qkv[1:q_size, :]
    k = qkv[q_size+1:q_size+kv_size, :]
    v = qkv[q_size+kv_size+1:end, :]
    
    if !isnothing(attn.q_bias); q .+= attn.q_bias; end
    if !isnothing(attn.k_bias); k .+= attn.k_bias; end
    if !isnothing(attn.v_bias); v .+= attn.v_bias; end
    
    # Reshape and RoPE
    q_r = reshape(q, attn.head_dim, attn.n_heads, seq)
    k_r = reshape(k, attn.head_dim, attn.n_kv_heads, seq)
    v_r = reshape(v, attn.head_dim, attn.n_kv_heads, seq)
    
    q_r = rope(q_r, pos)
    k_r = rope(k_r, pos)
    
    # Update KV Cache
    update!(kv_cache, k_r, v_r, pos)
    
    return x # Placeholder: Identity for scaffold
end

# --- Decoder Layer ---
struct DecoderLayer
    attn_norm::RMSNorm
    attn::Attention
    ssm::Union{Nothing, SSM}
    post_norm::RMSNorm
    mlp::MLP
end

function (layer::DecoderLayer)(x::oneMatrix{Float32}, pos::Int, rope::RotaryEmbedding, kv_cache::KVCache)
    # Residual: Attention
    identity = x
    x = layer.attn_norm(x)
    x = layer.attn(x, pos, rope, kv_cache)
    x = x .+ identity
    
    # Optional SSM for hybrid models
    if !isnothing(layer.ssm)
        identity = x
        x = layer.ssm(x)
        x = x .+ identity
    end
    
    # Residual: MLP
    identity = x
    x = layer.post_norm(x)
    x = layer.mlp(x)
    x = x .+ identity
    
    return x
end

# --- Qwen Model ---
struct QwenModel
    config::QwenConfig
    embed::oneArray{Float32, 2}
    layers::Vector{DecoderLayer}
    final_norm::RMSNorm
    lm_head::oneArray{Float32, 2}
    rope::RotaryEmbedding
end

function forward!(model::QwenModel, tokens::Vector{Int}, pos::Int, kv_caches::Vector{KVCache})
    # Embed
    # model.embed is (vocab_size, hidden_size) or (hidden_size, vocab_size)?
    # GGUF token_embd.weight is (hidden, vocab)
    x = model.embed[:, tokens] |> oneArray # (hidden, seq)
    
    for (i, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, kv_caches[i])
    end
    
    x = model.final_norm(x)
    logits = (x' * model.lm_head)' # (vocab, seq)
    
    return logits |> collect # Back to CPU
end

end # module
