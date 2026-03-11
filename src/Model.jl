module Model

using oneAPI
using LinearAlgebra
using Statistics

export QwenConfig, QwenModel, KVCache, forward!, RMSNorm, MLP, MambaBlock, FullAttention, DecoderLayer, init_kv_cache

# --- Configuration ---
Base.@kwdef struct QwenConfig
    vocab_size::Int = 151936
    hidden_size::Int = 1024
    intermediate_size::Int = 3584
    num_hidden_layers::Int = 24
    num_attention_heads::Int = 16   # q heads
    num_key_value_heads::Int = 2    # kv heads
    head_dim::Int = 256             # from attn_key_length
    rms_norm_eps::Float32 = 1e-6
    rope_theta::Float32 = 10000000.0
    max_position_embeddings::Int = 4096
    full_attention_interval::Int = 4  # every 4th layer is full attention
    ssm_inner_size::Int = 2048
    ssm_state_size::Int = 128
    ssm_group_count::Int = 16
end

const oneMatrix{T} = oneArray{T, 2}
const oneVector{T} = oneArray{T, 1}

# --- Normalization ---
struct RMSNorm
    weight::oneArray{Float16}
    eps::Float32
end

function (norm::RMSNorm)(x::oneArray{Float16})
    x32 = Float32.(x)
    # RMS per column (hidden dim)
    rms = reshape(sqrt.(vec(mean(x32 .^ 2, dims=1)) .+ norm.eps), 1, :)
    normalized = x32 ./ rms
    w = reshape(Float32.(norm.weight), :, 1)
    return Float16.(normalized .* w)
end

# --- Stable Mat-Mul (accumulate in Float32) ---
# weight: (K, N), x: (K, S) -> (N, S)  (weight is stored transposed in GGUF)
function mat_mul(weight::AbstractArray{Float16,2}, x::AbstractArray{Float16,2})
    K, N = size(weight)
    K_x, S = size(x)

    x_in = if K_x == K
        x
    elseif K_x < K
        p = zeros(Float16, K, S) |> oneArray
        @views p[1:K_x, :] .= x
        p
    else
        @view x[1:K, :]
    end

    chunk = 512
    res = zeros(Float16, N, S) |> oneArray
    for s in 1:S
        xv = @view x_in[:, s]
        for c in 1:chunk:N
            c_end = min(c + chunk - 1, N)
            w_c = @view weight[:, c:c_end]
            v32 = sum(Float32.(w_c) .* Float32.(xv), dims=1)
            @views res[c:c_end, s] .= vec(Float16.(v32))
        end
    end
    return res
end

# A: (M, N), B: (N, S) -> (M, S)
function mat_mul_AB(A::AbstractArray{Float16,2}, B::AbstractArray{Float16,2})
    M, N = size(A)
    N_b, S = size(B)
    chunk = 512
    res = zeros(Float16, M, S) |> oneArray
    for s in 1:S
        bv = @view B[:, s]
        for r in 1:chunk:M
            r_end = min(r + chunk - 1, M)
            a_c = @view A[r:r_end, :]
            v32 = sum(Float32.(a_c) .* Float32.(bv)', dims=2)
            @views res[r:r_end, s] .= vec(Float16.(v32))
        end
    end
    return res
end

# --- Rotary Embedding (RoPE) ---
struct RotaryEmbedding
    dim::Int
    base::Float32
    inv_freq::oneVector{Float32}
end

function RotaryEmbedding(dim::Int; base=10000000.0)
    inv_freq = 1.0f0 ./ (Float32(base) .^ (range(0, stop=dim - 1, step=2) ./ dim))
    return RotaryEmbedding(dim, Float32(base), oneArray(inv_freq))
end

function (rope::RotaryEmbedding)(x::oneArray{Float16,3}, pos::Int)
    d, h, seq = size(x)
    d_rope = min(d, rope.dim)
    positions = oneArray(Float32.(pos:(pos+seq-1)))
    freqs = rope.inv_freq[1:(d_rope÷2)] * positions'
    cos_t = reshape(Float16.(cos.(freqs)), d_rope ÷ 2, 1, seq)
    sin_t = reshape(Float16.(sin.(freqs)), d_rope ÷ 2, 1, seq)
    xr = reshape(@view(x[1:d_rope, :, :]), 2, d_rope ÷ 2, h, seq)
    x1 = @view xr[1, :, :, :]
    x2 = @view xr[2, :, :, :]
    res = similar(xr)
    res[1, :, :, :] = x1 .* cos_t .- x2 .* sin_t
    res[2, :, :, :] = x1 .* sin_t .+ x2 .* cos_t
    out = copy(x)
    out[1:d_rope, :, :] = reshape(res, d_rope, h, seq)
    return out
end

# --- KV Cache ---
mutable struct KVCache
    k::oneArray{Float16,3}  # (head_dim, n_kv, max_seq)
    v::oneArray{Float16,3}
    pos::Int
end

function init_kv_cache(head_dim, n_kv, max_seq)
    k = zeros(Float16, head_dim, n_kv, max_seq) |> oneArray
    v = zeros(Float16, head_dim, n_kv, max_seq) |> oneArray
    return KVCache(k, v, 0)
end

function update_kv!(cache::KVCache, k::oneArray{Float16,3}, v::oneArray{Float16,3})
    seq = size(k, 3)
    pos = cache.pos
    dk = min(size(cache.k, 1), size(k, 1))
    nk = min(size(cache.k, 2), size(k, 2))
    @views cache.k[1:dk, 1:nk, pos+1:pos+seq] .= k[1:dk, 1:nk, :]
    @views cache.v[1:dk, 1:nk, pos+1:pos+seq] .= v[1:dk, 1:nk, :]
    cache.pos += seq
end

# --- MLP ---
struct MLP
    gate::oneMatrix{Float16}
    up::oneMatrix{Float16}
    down::oneMatrix{Float16}
end

function (mlp::MLP)(x::oneMatrix{Float16})
    g = mat_mul(mlp.gate, x)
    u = mat_mul(mlp.up, x)
    g32 = Float32.(g)
    g .= Float16.(g32 ./ (1.0f0 .+ exp.(-g32)))  # silu
    return mat_mul(mlp.down, g .* u)
end

# --- Full Attention Layer ---
struct FullAttention
    q_weight::oneMatrix{Float16}
    k_weight::oneMatrix{Float16}
    v_weight::oneMatrix{Float16}
    o_weight::oneMatrix{Float16}
    q_norm::RMSNorm
    k_norm::RMSNorm
    n_heads::Int    # query heads
    n_kv::Int       # kv heads
    head_dim::Int
end

function (attn::FullAttention)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    h_dim, seq = size(x)
    # Project
    q = mat_mul(attn.q_weight, x)   # (n_heads * head_dim, seq)
    k = mat_mul(attn.k_weight, x)   # (n_kv * head_dim, seq)
    v = mat_mul(attn.v_weight, x)   # (n_kv * head_dim, seq)

    q_hd = attn.head_dim
    kv_hd = attn.head_dim

    # RMSNorm on each head
    qr3 = reshape(q, q_hd, attn.n_heads, seq)
    kr3 = reshape(k, kv_hd, attn.n_kv, seq)
    vr3 = reshape(v, kv_hd, attn.n_kv, seq)

    # Per-head norm (reshape to (head_dim, n_heads*seq), norm, reshape back)
    qn = reshape(attn.q_norm(reshape(qr3, q_hd, :)), q_hd, attn.n_heads, seq)
    kn = reshape(attn.k_norm(reshape(kr3, kv_hd, :)), kv_hd, attn.n_kv, seq)

    # RoPE
    qr = rope(qn, pos)
    kr = rope(kn, pos)

    update_kv!(cache, kr, vr3)

    total_len = cache.pos
    k_full = @view cache.k[:, :, 1:total_len]
    v_full = @view cache.v[:, :, 1:total_len]

    res_out = zeros(Float16, q_hd, attn.n_heads, seq) |> oneArray
    kv_per_q = attn.n_heads ÷ attn.n_kv  # GQA groups

    for h in 1:attn.n_heads
        kv_h = (h - 1) ÷ kv_per_q + 1
        qh = oneArray(collect(Float16, @view qr[:, h, :]))      # (head_dim, seq) on GPU
        kh = oneArray(collect(Float16, @view k_full[:, kv_h, :]))  # (head_dim, total_len), on GPU
        vh = oneArray(collect(Float16, @view v_full[:, kv_h, :]))  # (head_dim, total_len) on GPU

        # scores: (total_len, seq)  by treating kh as weight (K=head_dim) and qh as input
        scale = Float16(1.0f0 / sqrt(Float32(q_hd)))
        scores = mat_mul(kh, qh) .* scale

        # Causal mask
        for s in 1:seq
            mask_start = pos + s
            if mask_start < total_len
                @views scores[mask_start+1:total_len, s] .= Float16(-65504)
            end
        end

        s32 = Float32.(scores)
        mx = maximum(s32, dims=1)
        ex = exp.(s32 .- mx)
        pb = Float16.(ex ./ sum(ex, dims=1))   # (total_len, seq)
        res_out[:, h, :] = mat_mul_AB(vh, pb)  # (head_dim, seq)
    end

    return mat_mul(attn.o_weight, reshape(res_out, :, seq))
end

# --- Mamba (SSM) Block ---
# Qwen3.5 SSM hybrid: uses attn_qkv as in_proj, ssm_* params
struct MambaBlock
    in_proj::oneMatrix{Float16}   # (hidden, 6144): projects x -> [z(2048), x_expanded(2048), dt_base(ssm_groups)]
    gate_proj::oneMatrix{Float16} # (hidden, 2048): gate
    ssm_out::oneMatrix{Float16}   # (ssm_inner_size, hidden)
    ssm_a::oneVector{Float32}     # (ssm_groups,) log-scale A param
    ssm_alpha::oneMatrix{Float16} # (hidden, ssm_groups)
    ssm_beta::oneMatrix{Float16}  # (hidden, ssm_groups)
    ssm_conv1d::oneMatrix{Float32} # (conv_kernel, ssm_inner*2)  -- kept F32
    ssm_dt_bias::oneVector{Float32} # (ssm_groups,)
    ssm_norm::RMSNorm
    ssm_groups::Int   # = group_count = 16
    ssm_d::Int        # = state_size = 128
    inner_size::Int   # = 2048
end

function silu32(x::AbstractArray{T}) where T
    x32 = Float32.(x)
    return T.(x32 ./ (1.0f0 .+ exp.(-x32)))
end

function (m::MambaBlock)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    hidden, seq = size(x)

    # in_proj: (hidden, 6144) -> proj: (6144, seq)
    # Qwen3.5 split: 6144 = 2048(z_gate) + 2048(x_ssm) + 2048(dt_proj)
    proj = mat_mul(m.in_proj, x)

    inner = m.inner_size  # 2048

    z_gate = @view proj[1:inner, :]          # (2048, seq) - z gate
    x_ssm  = @view proj[inner+1:2*inner, :]  # (2048, seq) - SSM input

    # Additional gate from gate_proj
    gate_out = mat_mul(m.gate_proj, x)  # (2048, seq)
    gate = silu32(gate_out)

    # Gate the SSM input with SiLU-gated z
    z_act = silu32(z_gate)  # SiLU(z)
    gated_x = Float16.(Float32.(x_ssm) .* Float32.(z_act) .* Float32.(gate))

    # Output projection: (hidden, seq) from gated inner representation
    return mat_mul(m.ssm_out, gated_x)
end

# --- Decoder Layer ---
struct DecoderLayer
    in_norm::RMSNorm
    op::Union{MambaBlock, FullAttention}
    post_norm::RMSNorm
    mlp::MLP
    is_ssm::Bool
end

function (layer::DecoderLayer)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    h = layer.in_norm(x)
    h = layer.op(h, pos, rope, cache)
    x = x .+ h
    h = layer.post_norm(x)
    h = layer.mlp(h)
    x = x .+ h
    return x
end

# --- Model ---
struct QwenModel
    config::QwenConfig
    embed::oneMatrix{Float16}
    layers::Vector{DecoderLayer}
    final_norm::RMSNorm
    lm_head::oneMatrix{Float16}
    rope::RotaryEmbedding
end

function forward!(model::QwenModel, tokens::Vector{Int}, pos::Int, caches::Vector{KVCache})
    x = model.embed[:, tokens]              # (hidden, seq)
    for (i, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, caches[i])
    end
    x = model.final_norm(x)
    logits = mat_mul(model.lm_head, x)     # (vocab, seq)
    return Float32.(logits) |> collect
end

end # module
