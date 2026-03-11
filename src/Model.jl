module Model

using oneAPI
using LinearAlgebra
using Statistics

export QwenConfig, QwenModel, KVCache, forward!, RMSNorm, MLP, HybridBlock, Attention, DecoderLayer, init_kv_cache

# --- Configuration ---
Base.@kwdef struct QwenConfig
    vocab_size::Int = 151936
    hidden_size::Int = 1024
    intermediate_size::Int = 3584
    num_hidden_layers::Int = 24
    num_attention_heads::Int = 8
    num_key_value_heads::Int = 2
    head_dim::Int = 128
    rms_norm_eps::Float32 = 1e-6
    rope_theta::Float32 = 10000000.0
    max_position_embeddings::Int = 4096
end

# --- Normalization ---
struct RMSNorm
    weight::oneVector{Float16}
    eps::Float32
end

function (norm::RMSNorm)(x::oneMatrix{Float16})
    x32 = Float32.(x)
    # Correct RMS calculation: mean(x^2)
    rms = sqrt.(vec(mean(x32 .^ 2, dims=1)) .+ norm.eps)
    res32 = (x32 ./ rms') .* Float32.(norm.weight)
    return Float16.(res32)
end

# --- CHUNKED STABLE Mat-Mul ---
# This version avoids large broadcasts by chunking the output dimension.
# mat_mul_chunked computes weight' * x where weight: (K,N), x: (K,S) -> out: (N,S)
function mat_mul_chunked(weight::AbstractMatrix{Float16}, x::AbstractMatrix{Float16})
    # weight: (K, N), x: (K, S) -> out: (N, S)
    K, N = size(weight)
    K_x, S = size(x)

    # Handle padding/slicing for stability
    x_in = if K_x == K
        x
    elseif K_x < K
        p = zeros(Float16, K, S) |> oneArray
        @views p[1:K_x, :] .= x
        p
    else
        @view x[1:K, :]
    end

    chunk_size = 1024 # Small chunks for stability
    res = zeros(Float16, N, S) |> oneArray

    # Process tokens one-by-one for decode stability
    for s in 1:S
        xv = @view x_in[:, s]
        for c in 1:chunk_size:N
            c_end = min(c + chunk_size - 1, N)
            w_c = @view weight[:, c:c_end]
            # w_c: (K, chunk), xv: (K)
            # sum(w_c .* xv, dims=1) -> (1, chunk)
            res[c:c_end, s] = vec(sum(w_c .* xv, dims=1))
        end
    end
    return res
end

# mat_mul_chunked_AB computes A * B where A: (M, N), B: (N, S) -> out: (M, S)
# This is a companion to mat_mul_chunked that avoids large BLAS calls by chunking
# the output rows (M). Implemented similarly to mat_mul_chunked but for non-transposed.
function mat_mul_chunked_AB(A::AbstractMatrix{Float16}, B::AbstractMatrix{Float16})
    M, N = size(A)
    N_b, S = size(B)
    @assert N == N_b "Inner dimensions must match for matmul"

    chunk_size = 1024
    res = zeros(Float16, M, S) |> oneArray

    for s in 1:S
        bv = @view B[:, s]               # (N)
        # Broadcast multiply over each output-row chunk
        for r in 1:chunk_size:M
            r_end = min(r + chunk_size - 1, M)
            a_chunk = @view A[r:r_end, :]   # (chunk, N)
            # We want each row dot bv: sum(a_chunk .* bv', dims=2)
            # bv' is (1, N) and will broadcast across rows
            @views res[r:r_end, s] .= vec(sum(a_chunk .* bv', dims=2))
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
    # x: (dim, n_heads, seq)
    d, h, seq = size(x)
    d_rope = min(d, rope.dim)

    # Pre-calculate positions
    positions = oneArray(Float32.(pos:(pos+seq-1)))
    # freq: (dim/2, seq)
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
    k::oneArray{Float16,3}
    v::oneArray{Float16,3}
end

function init_kv_cache(head_dim, n_kv, max_seq)
    k = zeros(Float16, head_dim, n_kv, max_seq) |> oneArray
    v = zeros(Float16, head_dim, n_kv, max_seq) |> oneArray
    return KVCache(k, v)
end

function update!(cache::KVCache, k::oneArray{Float16,3}, v::oneArray{Float16,3}, pos::Int)
    seq = size(k, 3)
    d, n, _ = size(cache.k)
    # Safe copy with slicing
    dk = min(d, size(k, 1))
    nk = min(n, size(k, 2))

    @views cache.k[1:dk, 1:nk, pos+1:pos+seq] .= k[1:dk, 1:nk, :]
    @views cache.v[1:dk, 1:nk, pos+1:pos+seq] .= v[1:dk, 1:nk, :]
end

# --- MLP ---
struct MLP
    gate::oneArray{Float16,2}
    up::oneArray{Float16,2}
    down::oneArray{Float16,2}
end

function (mlp::MLP)(x::oneMatrix{Float16})
    g = mat_mul_chunked(mlp.gate, x)
    u = mat_mul_chunked(mlp.up, x)
    # Silu/Swish: x * sigmoid(x)
    g32 = Float32.(g)
    g .= Float16.(g32 .* (1.0f0 ./ (1.0f0 .+ exp.(-g32))))
    return mat_mul_chunked(mlp.down, g .* u)
end

# --- Attention ---
struct Attention
    q_weight::oneArray{Float16,2}
    k_weight::oneArray{Float16,2}
    v_weight::oneArray{Float16,2}
    o_weight::oneArray{Float16,2}
    n_heads::Int
    n_kv::Int
end

function (attn::Attention)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    h_dim, seq = size(x)
    # q, k, v projections
    q = mat_mul_chunked(attn.q_weight, x)   # q = q_weight' * x  -> (q_dim, seq)
    k = mat_mul_chunked(attn.k_weight, x)   # k = k_weight' * x  -> (kv_dim, seq)
    v = mat_mul_chunked(attn.v_weight, x)   # v = v_weight' * x  -> (kv_dim, seq)

    q_hd = size(q, 1) ÷ attn.n_heads
    kv_hd = size(k, 1) ÷ attn.n_kv

    qr = rope(reshape(q, q_hd, attn.n_heads, seq), pos)
    kr = rope(reshape(k, kv_hd, attn.n_kv, seq), pos)
    vr = reshape(v, kv_hd, attn.n_kv, seq)

    update!(cache, kr, vr, pos)

    # Use full context for attention
    total_len = pos + seq
    k_full = @view cache.k[:, :, 1:total_len]
    v_full = @view cache.v[:, :, 1:total_len]

    # attn_out: (head_dim, n_heads, seq)
    res_out = zeros(Float16, q_hd, attn.n_heads, seq) |> oneArray

    # Head-by-head loop for B580 stability (prevents large GPU-wide broadcasts)
    for h in 1:attn.n_heads
        kv_h = ((h - 1) * attn.n_kv ÷ attn.n_heads) + 1
        qh = @view qr[:, h, :]        # (q_hd, seq)
        kh = @view k_full[:, kv_h, :] # (kv_hd, total_len)
        vh = @view v_full[:, kv_h, :] # (kv_hd, total_len)

        # Align query/key dims for MLA: use smallest common dimension so multiplication shapes match.
        use_dim = min(size(qh, 1), size(kh, 1))
        qh_r = @view qh[1:use_dim, :]  # (use_dim, seq)
        kh_r = @view kh[1:use_dim, :]  # (use_dim, total_len)
        vh_r = @view vh[1:use_dim, :]  # (use_dim, total_len)

        # scores: (total_len, seq)
        # Replace direct BLAS GEMM with safe chunked transpose-matmul to avoid driver hangs.
        # mat_mul_chunked computes weight' * x for weight:(K,N), x:(K,S) -> (N,S).
        # Passing kh_r (K=use_dim, N=total_len) and qh_r (K=use_dim, S=seq) yields kh_r' * qh_r.
        scores = mat_mul_chunked(kh_r, qh_r) ./ Float16(sqrt(use_dim))

        # Causal masking for prefill
        if seq > 1
            for s in 1:seq
                mask_end = pos + s
                if mask_end < total_len
                    @views scores[(mask_end+1):total_len, s] .= -65500.0f0
                end
            end
        end

        # Softmax per column (token)
        scores_32 = Float32.(scores)
        mx = maximum(scores_32, dims=1)
        ex = exp.(scores_32 .- mx)
        pb = Float16.(ex ./ sum(ex, dims=1))   # (total_len, seq)

        # heads_res: (use_dim, seq) computed safely without large GEMM
        # vh_r: (use_dim, total_len), pb: (total_len, seq) -> result (use_dim, seq)
        heads_res = mat_mul_chunked_AB(vh_r, pb)

        # Write into res_out. If the query head-dim (q_hd) is larger than use_dim,
        # leave the remaining slots zero (effectively zero-padding). This keeps the
        # output tensor shape consistent while supporting MLA-style reduced KV dims.
        @views res_out[1:use_dim, h, :] .= heads_res
    end

    return mat_mul_chunked(attn.o_weight, reshape(res_out, :, seq))
end

# --- Hybrid Block (SSM) ---
struct HybridBlock
    qkv_weight::oneArray{Float16,2}
    gate_weight::oneArray{Float16,2}
    out_weight::oneArray{Float16,2}
end

function (block::HybridBlock)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    p = mat_mul_chunked(block.qkv_weight, x)
    g = mat_mul_chunked(block.gate_weight, x)

    # Functional approximation of gated SSM
    g32 = 1.0f0 ./ (1.0f0 .+ exp.(-Float32.(g)))
    # We slice qkv to match gate size if needed
    p_use = size(p, 1) > size(g, 1) ? (@view p[1:size(g, 1), :]) : p

    res = Float16.(g32) .* p_use
    return mat_mul_chunked(block.out_weight, res)
end

# --- Decoder Layer ---
struct DecoderLayer
    in_norm::RMSNorm
    op::Union{HybridBlock,Attention}
    post_norm::RMSNorm
    mlp::MLP
end

function (layer::DecoderLayer)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    h = layer.in_norm(x)
    h = layer.op(h, pos, rope, cache)
    x = x .+= h

    h = layer.post_norm(x)
    h = layer.mlp(h)
    x = x .+= h
    return x
end

# --- Model ---
struct QwenModel
    config::QwenConfig
    embed::oneArray{Float16,2}
    layers::Vector{DecoderLayer}
    final_norm::RMSNorm
    lm_head::oneArray{Float16,2}
    rope::RotaryEmbedding
end

function forward!(model::QwenModel, tokens::Vector{Int}, pos::Int, caches::Vector{KVCache})
    # tokens are 1-indexed token IDs
    x = model.embed[:, tokens]
    for (i, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, caches[i])
    end
    x = model.final_norm(x)
    logits = mat_mul_chunked(model.lm_head, x)
    return Float32.(logits) |> collect
end

end # module
