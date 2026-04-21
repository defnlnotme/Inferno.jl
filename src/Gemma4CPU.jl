module Gemma4CPU

using ..ModelCPU
using ..Tokenizer
using LinearAlgebra
using Printf

# ============================================================
# RoPE (Rotary Position Embeddings) for Gemma4
# Two configs: sliding (full rotary, theta=10000) and 
# full (partial 25% rotary, theta=1M, proportional)
# ============================================================

function precompute_freqs_cis(head_dim::Int, max_seq_len::Int, theta::Float64; partial_rotary_factor::Float64=1.0)
    rotary_dim = Int(head_dim * partial_rotary_factor)
    freqs = Float64[1.0 / (theta ^ (2k / rotary_dim)) for k in 0:(rotary_dim÷2 - 1)]
    t = Float64[Float64(i) for i in 0:(max_seq_len - 1)]
    freqs_t = t .* freqs'  # [seq_len, rotary_dim/2]
    cos_freqs = cos.(freqs_t)
    sin_freqs = sin.(freqs_t)
    return Float32.(cos_freqs), Float32.(sin_freqs)
end

function apply_rope!(q::Vector{Float32}, cos_f::Float32, sin_f::Float32, head_dim::Int, partial_rotary_factor::Float64=1.0)
    rotary_dim = Int(head_dim * partial_rotary_factor)
    half = rotary_dim ÷ 2
    for k in 0:(half - 1)
        q0 = q[k + 1]
        q1 = q[k + half + 1]
        q[k + 1]       = q0 * cos_f - q1 * sin_f
        q[k + half + 1] = q0 * sin_f + q1 * cos_f
    end
end

function apply_rope_kv!(k::Vector{Float32}, cos_f::Float32, sin_f::Float32, head_dim::Int, partial_rotary_factor::Float64=1.0)
    apply_rope!(k, cos_f, sin_f, head_dim, partial_rotary_factor)
end

# ============================================================
# RMSNorm (Gemma4 variant: weight * norm, no +1 bias)
# ============================================================

function rmsnorm_g4!(out::Vector{Float32}, x::Vector{Float32}, w::Vector{Float32}, eps::Float32)
    n = length(x)
    s = eps
    @simd for i in 1:n
        s += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(s / n)
    @simd for i in 1:n
        out[i] = w[i] * x[i] * inv_rms
    end
end

function rmsnorm_g4_noscale!(out::Vector{Float32}, x::Vector{Float32}, eps::Float32)
    n = length(x)
    s = eps
    @simd for i in 1:n
        s += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(s / n)
    @simd for i in 1:n
        out[i] = x[i] * inv_rms
    end
end

# ============================================================
# GELU Tanh Activation
# ============================================================

function gelu_tanh(x::Float32)
    c = 0.7978845608028654f0  # sqrt(2/pi)
    inner = c * (x + 0.044715f0 * x * x)
    return 0.5f0 * x * (1.0f0 + tanh(inner))
end

# ============================================================
# KV Cache
# ============================================================

struct KVCacheG4
    k::Matrix{Float32}  # [max_seq_len, num_kv_heads * head_dim]
    v::Matrix{Float32}
    len::Vector{Int}     # current length (mutable ref)
end

function init_kv_cache_g4(max_len::Int, num_kv_heads::Int, head_dim::Int)
    KVCacheG4(
        zeros(Float32, max_len, num_kv_heads * head_dim),
        zeros(Float32, max_len, num_kv_heads * head_dim),
        [0]
    )
end

# ============================================================
# Attention Layer
# ============================================================

mutable struct Gemma4AttentionCPU
    layer_idx::Int
    layer_type::Symbol  # :sliding or :full
    is_sliding::Bool
    sliding_window::Int
    head_dim::Int
    num_q_heads::Int
    num_kv_heads::Int
    num_kv_groups::Int
    partial_rotary_factor::Float64
    
    # Weights (all stored as F32 matrices, transposed for BLAS)
    q_proj::Matrix{Float32}  # [hidden, num_q_heads * head_dim]
    k_proj::Union{Matrix{Float32}, Nothing}  # nothing if kv_shared
    v_proj::Union{Matrix{Float32}, Nothing}
    o_proj::Matrix{Float32}  # [num_q_heads * head_dim, hidden]
    
    # Norms
    q_norm_w::Vector{Float32}
    k_norm_w::Union{Vector{Float32}, Nothing}
    v_norm_w::Union{Vector{Float32}, Nothing}  # noscale
    
    # KV sharing
    is_kv_shared::Bool
    kv_shared_layer_idx::Union{Int, Nothing}
    store_full_kv::Bool
    
    # Buffers
    q_buf::Vector{Float32}
    k_buf::Vector{Float32}
    v_buf::Vector{Float32}
    attn_buf::Vector{Float32}
    q_normed::Vector{Float32}
    k_normed::Vector{Float32}
    v_normed::Vector{Float32}
end

# ============================================================
# MLP Layer
# ============================================================

mutable struct Gemma4MLPCPU
    hidden_size::Int
    intermediate_size::Int
    gate_proj::Matrix{Float32}
    up_proj::Matrix{Float32}
    down_proj::Matrix{Float32}
    
    # Buffers
    gate_buf::Vector{Float32}
    up_buf::Vector{Float32}
    intermediated::Vector{Float32}
end

# ============================================================
# Per-Layer Input
# ============================================================

mutable struct Gemma4PerLayerInputCPU
 hidden_size::Int
 per_layer_dim::Int
 num_layers::Int
 gate_proj::Matrix{Float32} # [hidden, per_layer_dim] (per-layer input gate)
 projection::Matrix{Float32} # [per_layer_dim, hidden] (per-layer projection)
 model_projection::Matrix{Float32} # [hidden, num_layers * per_layer_dim] (projects main embedding to per-layer space)
 model_projection_scale::Float32 # = 1/sqrt(hidden_size)
 per_layer_input_scale::Float32 # = 1/sqrt(2)
 post_norm_w::Vector{Float32} # per_layer_projection_norm weight [num_layers * per_layer_dim]
 
 # Buffers
 gate_buf::Vector{Float32}
 proj_buf::Vector{Float32}
 model_proj_buf::Vector{Float32} # for model_projection output
 combined_buf::Vector{Float32} # for combined per-layer input
end

# ============================================================
# Decoder Layer
# ============================================================

mutable struct Gemma4DecoderLayerCPU
    layer_idx::Int
    hidden_size::Int
    layer_type::Symbol
    attention::Gemma4AttentionCPU
    mlp::Gemma4MLPCPU
    
    # Norms
    input_norm_w::Vector{Float32}
    post_attn_norm_w::Vector{Float32}
    pre_ff_norm_w::Vector{Float32}
    post_ff_norm_w::Vector{Float32}
    layer_scalar::Float32
    
    # Per-layer input
    per_layer_input::Union{Gemma4PerLayerInputCPU, Nothing}
    
    # Norm buffers
    norm_buf1::Vector{Float32}
    norm_buf2::Vector{Float32}
    residual::Vector{Float32}
end

# ============================================================
# Full Model
# ============================================================

mutable struct Gemma4ModelCPU
    config::Dict{String, Any}
    hidden_size::Int
    num_layers::Int
    num_q_heads::Int
    num_kv_heads::Int
    head_dim::Int
    global_head_dim::Int
    vocab_size::Int
    sliding_window::Int
    rms_norm_eps::Float32
    logit_softcapping::Float32
    embed_scale::Float32
    per_layer_dim::Int
    
    # Embeddings
    embed_tokens::Matrix{Float32}  # [vocab_size, hidden_size]
    embed_per_layer::Union{Matrix{Float32}, Nothing}  # [vocab_per_layer, num_layers * per_layer_dim]
    final_norm_w::Vector{Float32}
    
    # Layers
    layers::Vector{Gemma4DecoderLayerCPU}
    layer_types::Vector{Symbol}
    
    # KV caches (one per non-shared layer)
    kv_caches::Vector{KVCacheG4}
    
    # Shared KV states (stored by "storing" layers)
    shared_kv_k::Dict{Int, Matrix{Float32}}
    shared_kv_v::Dict{Int, Matrix{Float32}}
    
    # RoPE tables
    sliding_cos::Matrix{Float32}
    sliding_sin::Matrix{Float32}
    full_cos::Matrix{Float32}
    full_sin::Matrix{Float32}
    
    # lm_head (tied with embed_tokens or separate)
    lm_head::Matrix{Float32}
    
    # Tokenizer
    tokenizer::Union{BPETokenizer, Nothing}
end

end # module
