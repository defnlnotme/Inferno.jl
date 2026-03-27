"""
CPU-only inference backend for Inferno.jl
This module provides pure CPU implementations without GPU dependencies.
"""
module ModelCPU

using LinearAlgebra
using Statistics

export QwenConfigCPU, QwenModelCPU, KVCacheCPU, forward_cpu!, RMSNormCPU, MLPCPU, GatedDeltaNetCPU, FullAttentionCPU, DecoderLayerCPU
export init_kv_cache_cpu, reset_states_cpu!

# --- Configuration ---
Base.@kwdef struct QwenConfigCPU
    architecture::Symbol = :qwen
    vocab_size::Int = 151936
    hidden_size::Int = 1024
    intermediate_size::Int = 3584
    num_hidden_layers::Int = 24
    num_attention_heads::Int = 8
    num_key_value_heads::Int = 2
    head_dim::Int = 256
    rms_norm_eps::Float32 = 1e-6f0
    rope_theta::Float32 = 10000000.0f0
    max_position_embeddings::Int = 4096
    full_attention_interval::Int = 4
    ssm_inner_size::Int = 2048
    ssm_state_size::Int = 128
    ssm_group_count::Int = 16
    ssm_time_step_rank::Int = 16
    ssm_conv_kernel::Int = 4
end

# --- Normalization ---
struct RMSNormCPU
    weight::Vector{Float32}
    eps::Float32
end

function RMSNormCPU(weight::AbstractArray{Float32}, eps::Float32)
    return RMSNormCPU(vec(weight), eps)
end

function (norm::RMSNormCPU)(x::AbstractArray{Float32})
    ss = mapreduce(abs2, +, x)
    m = ss / length(x)
    scale = 1.0f0 / sqrt(m + norm.eps)
    return x .* scale .* norm.weight
end

function rmsnorm_cpu!(out::AbstractArray{Float32}, x::AbstractArray{Float32}, norm::RMSNormCPU)
    ss = mapreduce(abs2, +, x)
    m = ss / length(x)
    scale = 1.0f0 / sqrt(m + norm.eps)
    out .= x .* scale .* norm.weight
    return out
end

# --- Rotary Position Embedding ---
struct RotaryEmbeddingCPU
    inv_freq::Vector{Float32}
    max_seq_len::Int
end

function RotaryEmbeddingCPU(head_dim::Int, theta::Float32 = 10000.0f0, max_seq_len::Int = 4096)
    inv_freq = Float32[1.0 / (theta ^ (2(i-1)/head_dim)) for i in 1:div(head_dim, 2)]
    return RotaryEmbeddingCPU(inv_freq, max_seq_len)
end

function apply_rotary_emb!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU)
    head_dim, num_heads = size(x, 1), size(x, 2)
    half = div(head_dim, 2)
    
    for h in 1:num_heads
        for i in 1:half
            freq = rope.inv_freq[i] * pos
            cos_val = cos(freq)
            sin_val = sin(freq)
            
            idx1 = i
            idx2 = i + half
            
            x1 = x[idx1, h]
            x2 = x[idx2, h]
            
            x[idx1, h] = x1 * cos_val - x2 * sin_val
            x[idx2, h] = x1 * sin_val + x2 * cos_val
        end
    end
    return x
end

# --- KV Cache ---
struct KVCacheCPU
    k::Array{Float32,3}  # (head_dim, n_kv_heads, max_seq)
    v::Array{Float32,3}
    pos::Int
end

function init_kv_cache_cpu(config::QwenConfigCPU, max_seq::Int = 4096)
    k = zeros(Float32, config.head_dim, config.num_key_value_heads, max_seq)
    v = zeros(Float32, config.head_dim, config.num_key_value_heads, max_seq)
    return KVCacheCPU(k, v, 0)
end

function update_kv_cache!(cache::KVCacheCPU, k::Matrix{Float32}, v::Matrix{Float32}, pos::Int)
    @views cache.k[:, :, pos + 1] .= k
    @views cache.v[:, :, pos + 1] .= v
    return cache
end

# --- MLP ---
struct MLPCPU
    gate_weight::Matrix{Float32}  # (intermediate, hidden)
    up_weight::Matrix{Float32}
    down_weight::Matrix{Float32}
end

function (mlp::MLPCPU)(x::Vector{Float32})
    # Gate with SiLU
    gate = mlp.gate_weight * x
    @. gate = gate * (1.0f0 / (1.0f0 + exp(-gate)))  # SiLU
    
    # Up projection
    up = mlp.up_weight * x
    
    # Element-wise multiply
    hidden = gate .* up
    
    # Down projection
    return mlp.down_weight * hidden
end

# --- GatedDeltaNet (SSM) ---
struct GatedDeltaNetCPU
    index::Int
    
    # Weights (transposed for matrix-vector multiply)
    in_proj::Matrix{Float32}     # (conv_channels, hidden)
    gate_proj::Matrix{Float32}   # (d_inner, hidden)
    ssm_out::Matrix{Float32}     # (hidden, d_inner)
    ssm_conv1d::Matrix{Float32}  # (conv_channels, conv_kernel)
    
    # Alpha/beta projections
    ssm_alpha_weight::Matrix{Float32}  # (num_v_heads, hidden)
    ssm_beta_weight::Matrix{Float32}
    
    # SSM parameters
    ssm_a::Vector{Float32}
    ssm_dt_bias::Vector{Float32}
    ssm_norm::RMSNormCPU
    
    # Dimensions
    num_v_heads::Int
    num_k_heads::Int
    head_k_dim::Int
    head_v_dim::Int
    d_inner::Int
    conv_channels::Int
    conv_kernel::Int
    
    # State buffers
    conv_state::Matrix{Float32}
    h::Array{Float32,3}
end

function reset_states_cpu!(m::GatedDeltaNetCPU)
    fill!(m.conv_state, 0.0f0)
    fill!(m.h, 0.0f0)
    return nothing
end

function (m::GatedDeltaNetCPU)(x::Vector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
    # 1. Input projections
    qkv = m.in_proj * x
    z = m.gate_proj * x
    
    # 2. Update conv state (ring buffer)
    if m.conv_kernel > 1
        m.conv_state[:, 1:(m.conv_kernel-1)] .= m.conv_state[:, 2:m.conv_kernel]
    end
    m.conv_state[:, m.conv_kernel] .= qkv
    
    # 3. Compute convolution
    x_conv = zeros(Float32, m.conv_channels)
    for k in 1:m.conv_kernel
        for c in 1:m.conv_channels
            x_conv[c] += m.conv_state[c, k] * m.ssm_conv1d[c, k]
        end
    end
    
    # 4. SiLU activation
    @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))
    
    # 5. Split into Q, K, V
    qk_size = m.head_k_dim * m.num_k_heads
    q_all = reshape(view(x_conv, 1:qk_size), m.head_k_dim, m.num_k_heads)
    k_all = reshape(view(x_conv, qk_size+1:2*qk_size), m.head_k_dim, m.num_k_heads)
    v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+m.d_inner), m.head_v_dim, m.num_v_heads)
    
    # 6. Alpha/beta projections
    alpha_proj = m.ssm_alpha_weight * x
    beta_proj = m.ssm_beta_weight * x
    
    # 7. Process each head (delta net)
    y_all = zeros(Float32, m.d_inner)
    scale = 1.0f0 / sqrt(Float32(m.head_k_dim))
    
    for h in 1:m.num_v_heads
        g = ((h - 1) % m.num_k_heads) + 1
        
        qg = view(q_all, :, g)
        kg = view(k_all, :, g)
        vg = view(v_all, :, h)
        
 # Q/K L2 normalization
 q_norm = sqrt(sum(abs2, qg) + Float32(1e-6))
 k_norm = sqrt(sum(abs2, kg) + Float32(1e-6))
        
        q_normalized = qg ./ q_norm .* scale
        k_normalized = kg ./ k_norm
        
        # Gate values with numerical stability
        alpha_val = clamp(Float64(alpha_proj[h]) + Float64(m.ssm_dt_bias[h]), -20.0, 20.0)
        softplus_alpha = log(1.0 + exp(alpha_val))
        softplus_alpha = clamp(softplus_alpha, -10.0, 10.0)
        
        decay = Float32(exp(softplus_alpha * Float64(m.ssm_a[h])))
        decay = clamp(decay, 0.0f0, 1.0f0)
        
        beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
        beta = Float32(1.0 / (1.0 + exp(-beta_val)))
        
        # State operations
        state = view(m.h, :, :, h)
        state .*= decay
        
        sk = state * k_normalized
        tmp_head = beta .* (vg .- sk)
        
        # Outer product: state += tmp_head * k_normalized'
        BLAS.ger!(1.0f0, tmp_head, k_normalized, state)
        
        # Output
        yg = view(y_all, (h-1)*m.head_v_dim+1:h*m.head_v_dim)
        mul!(yg, state, q_normalized)
    end
    
 # 8. Apply SSM norm (per-head normalization)
 # y_all has shape (d_inner,) = (head_v_dim * num_v_heads,)
 # norm is applied per-head
 for h in 1:m.num_v_heads
     y_h = view(y_all, (h-1)*m.head_v_dim+1:h*m.head_v_dim)
     rmsnorm_cpu!(y_h, y_h, m.ssm_norm)
 end
 
 # 9. SiLU gate on z
 @. y_all = y_all * z * (1.0f0 / (1.0f0 + exp(-z)))
 
 # 10. Output projection
 return m.ssm_out * y_all
end

# --- Full Attention ---
struct FullAttentionCPU
    index::Int
    wq::Matrix{Float32}
    wk::Matrix{Float32}
    wv::Matrix{Float32}
    wo::Matrix{Float32}
    q_norm::RMSNormCPU
    k_norm::RMSNormCPU
    n_heads::Int
    n_kv::Int
    head_dim::Int
    scale::Float32
end

function (attn::FullAttentionCPU)(x::Vector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
    # Q, K, V projections
    q = attn.wq * x
    k = attn.wk * x
    v = attn.wv * x
    
    # Reshape to (head_dim, num_heads)
    q = reshape(q, attn.head_dim, attn.n_heads)
    k = reshape(k, attn.head_dim, attn.n_kv)
    v = reshape(v, attn.head_dim, attn.n_kv)
    
    # Apply Q/K normalization
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    
    # Apply RoPE
    apply_rotary_emb!(q, pos, rope)
    apply_rotary_emb!(k, pos, rope)
    
    # Update KV cache
    update_kv_cache!(cache, k, v, pos)
    
    # Compute attention scores
    output = zeros(Float32, attn.head_dim * attn.n_heads)
    
    gqa_ratio = div(attn.n_heads, attn.n_kv)
    
    for h in 1:attn.n_heads
        kv_h = div(h - 1, gqa_ratio) + 1
        
        q_h = view(q, :, h)
        
        # Compute scores for all cached positions
        scores = zeros(Float32, pos + 1)
        for p in 0:pos
            k_h = view(cache.k, :, kv_h, p + 1)
            scores[p + 1] = dot(q_h, k_h) * attn.scale
        end
        
        # Softmax
        max_score = maximum(scores)
        scores = exp.(scores .- max_score)
        scores ./= sum(scores)
        
        # Weighted sum of values
        out_h = zeros(Float32, attn.head_dim)
        for p in 0:pos
            v_h = view(cache.v, :, kv_h, p + 1)
            out_h .+= scores[p + 1] .* v_h
        end
        
        output[(h-1)*attn.head_dim+1:h*attn.head_dim] .= out_h
    end
    
    # Output projection
    return attn.wo * output
end

# --- Decoder Layer ---
struct DecoderLayerCPU
    in_norm::RMSNormCPU
    op::Union{GatedDeltaNetCPU,FullAttentionCPU}
    post_norm::RMSNormCPU
    mlp::MLPCPU
    is_ssm::Bool
end

function (layer::DecoderLayerCPU)(x::AbstractVector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
    # Input normalization
    x_norm = layer.in_norm(x)
    
    # Attention/SSM
    residual = layer.op(x_norm, pos, rope, cache)
    
    # Residual connection
    x = x + residual
    
    # Post-attention normalization
    x_norm = layer.post_norm(x)
    
    # MLP
    residual = layer.mlp(x_norm)
    
    # Final residual
    return x + residual
end

# --- Full Model ---
struct QwenModelCPU
    config::QwenConfigCPU
    embed::Matrix{Float32}  # (hidden, vocab_size)
    lm_head::Matrix{Float32}  # (vocab_size, hidden) or tied to embed
    layers::Vector{DecoderLayerCPU}
    final_norm::RMSNormCPU
    rope::RotaryEmbeddingCPU
end

function forward_cpu!(model::QwenModelCPU, tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU})
    seq_len = length(tokens)
    all_logits = zeros(Float32, model.config.vocab_size, seq_len)
    
    for t in 1:seq_len
        tok = tokens[t]
        curr_pos = pos + t - 1
        
        # Get embedding
        x = view(model.embed, :, tok)  # (hidden,)
        
        # Process through layers
        for (i, layer) in enumerate(model.layers)
            x = layer(x, curr_pos, model.rope, caches[i])
        end
        
        # Final normalization
        x = model.final_norm(x)
        
        # LM head (compute logits)
        logits = model.lm_head * x
        all_logits[:, t] = logits
    end
    
    return all_logits
end

function reset_states_cpu!(model::QwenModelCPU)
    for layer in model.layers
        if layer.is_ssm
            reset_states_cpu!(layer.op)
        end
    end
end

end # module
