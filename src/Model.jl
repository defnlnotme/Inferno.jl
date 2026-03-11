module Model

using oneAPI
using LinearAlgebra
using Statistics

export QwenConfig, QwenModel, KVCache, forward!, RMSNorm, MLP, GatedDeltaNet, FullAttention, DecoderLayer, init_kv_cache

# --- Configuration ---
Base.@kwdef struct QwenConfig
    vocab_size::Int = 151936
    hidden_size::Int = 1024
    intermediate_size::Int = 3584
    num_hidden_layers::Int = 24
    num_attention_heads::Int = 8    # q heads for full-attention layers
    num_key_value_heads::Int = 2    # kv heads for full-attention layers
    head_dim::Int = 256             # from attn_key_length
    rms_norm_eps::Float32 = 1e-6
    rope_theta::Float32 = 10000000.0
    max_position_embeddings::Int = 4096
    full_attention_interval::Int = 4
    ssm_inner_size::Int = 2048
    ssm_state_size::Int = 128       # head_k_dim
    ssm_group_count::Int = 16       # num_k_heads = num_v_heads
    ssm_time_step_rank::Int = 16    # num_v_heads
end

const oneMatrix{T} = oneArray{T, 2}
const oneVector{T} = oneArray{T, 1}

# --- Normalization ---
struct RMSNorm
    weight::oneArray{Float16}
    eps::Float32
end

function (norm::RMSNorm)(x::AbstractArray{Float16})
    x32 = Float32.(x)
    rms = reshape(sqrt.(vec(mean(x32 .^ 2, dims=1)) .+ norm.eps), 1, :)
    normalized = x32 ./ rms
    w = reshape(Float32.(norm.weight), :, 1)
    return Float16.(normalized .* w)
end

# --- Stable Mat-Mul (accumulate in Float32) ---
# weight: (K, N), x: (K, S) -> (N, S)  — weight stored transposed in GGUF
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
    chunk = 512
    S = size(B, 2)
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

# --- SiLU ---
function silu_f16(x::AbstractArray{Float16})
    x32 = Float32.(x)
    Float16.(x32 ./ (1.0f0 .+ exp.(-x32)))
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
    g = silu_f16(g)
    return mat_mul(mlp.down, g .* u)
end

# --- Full Attention Layer ---
# Q weight packs both Q and gate interleaved: attn_q output is (head_dim*2*n_heads, seq)
# Q = output[0:head_dim, h, :], gate = output[head_dim:2*head_dim, h, :]
struct FullAttention
    q_weight::oneMatrix{Float16}   # (hidden, head_dim*2*n_heads)
    k_weight::oneMatrix{Float16}   # (hidden, head_dim*n_kv)
    v_weight::oneMatrix{Float16}   # (hidden, head_dim*n_kv)
    o_weight::oneMatrix{Float16}   # (head_dim*n_heads, hidden)
    q_norm::RMSNorm                # (head_dim,) per-head Q norm
    k_norm::RMSNorm                # (head_dim,) per-head K norm
    n_heads::Int
    n_kv::Int
    head_dim::Int
end

function (attn::FullAttention)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    h_dim, seq = size(x)
    hd = attn.head_dim  # 256

    # attn_q output: (head_dim*2*n_heads, seq) — packed Q+gate
    q_full = mat_mul(attn.q_weight, x)  # (hd*2*n_heads, seq)

    # Split into Q and gate by extracting alternating hd-size blocks
    # Q_full layout: [Q_h0(hd), gate_h0(hd), Q_h1(hd), gate_h1(hd), ...]
    q_3d = reshape(q_full, hd * 2, attn.n_heads, seq)
    q_only = q_3d[1:hd, :, :]       # (hd, n_heads, seq)
    gate_raw = q_3d[hd+1:2*hd, :, :] # (hd, n_heads, seq)

    # K, V projections
    k = mat_mul(attn.k_weight, x)  # (hd*n_kv, seq)
    v = mat_mul(attn.v_weight, x)  # (hd*n_kv, seq)

    kr3 = reshape(k, hd, attn.n_kv, seq)
    vr3 = reshape(v, hd, attn.n_kv, seq)

    # Per-head Q/K norms
    qn = reshape(attn.q_norm(reshape(q_only, hd, :)), hd, attn.n_heads, seq)
    kn = reshape(attn.k_norm(reshape(kr3, hd, :)), hd, attn.n_kv, seq)

    # RoPE
    qr = rope(qn, pos)
    kr_rope = rope(kn, pos)

    update_kv!(cache, kr_rope, vr3)

    total_len = cache.pos
    k_full = @view cache.k[:, :, 1:total_len]
    v_full = @view cache.v[:, :, 1:total_len]

    res_out = zeros(Float16, hd, attn.n_heads, seq) |> oneArray
    kv_per_q = attn.n_heads ÷ attn.n_kv

    for h in 1:attn.n_heads
        kv_h = (h - 1) ÷ kv_per_q + 1
        qh = oneArray(collect(Float16, @view qr[:, h, :]))
        kh = oneArray(collect(Float16, @view k_full[:, kv_h, :]))
        vh = oneArray(collect(Float16, @view v_full[:, kv_h, :]))

        scale = Float16(1.0f0 / sqrt(Float32(hd)))
        scores = mat_mul(kh, qh) .* scale  # (total_len, seq)

        for s in 1:seq
            mask_start = pos + s
            if mask_start < total_len
                @views scores[mask_start+1:total_len, s] .= Float16(-65504)
            end
        end

        s32 = Float32.(scores)
        mx = maximum(s32, dims=1)
        ex = exp.(s32 .- mx)
        pb = Float16.(ex ./ sum(ex, dims=1))
        res_out[:, h, :] = mat_mul_AB(vh, pb)
    end

    # Apply sigmoid gate: attn_out * sigmoid(gate)
    # gate_raw: (hd, n_heads, seq) — flatten to (hd*n_heads, seq) for sigmoid
    gate_2d = reshape(gate_raw, hd * attn.n_heads, seq)
    gate_sig = Float16.(1.0f0 ./ (1.0f0 .+ exp.(Float32.(.-gate_2d))))  # sigmoid
    gated_out = reshape(res_out, hd * attn.n_heads, seq) .* gate_sig

    return mat_mul(attn.o_weight, gated_out)
end

# --- Gated Delta Net (SSM Layer) ---
# Reference: qwen35.cpp build_layer_attn_linear
struct GatedDeltaNet
    in_proj::oneMatrix{Float16}     # wqkv: (hidden, 6144) — projects to qkv_mixed
    gate_proj::oneMatrix{Float16}   # wqkv_gate: (hidden, d_inner=2048) — projects to z
    ssm_out::oneMatrix{Float16}     # (d_inner, hidden)
    ssm_a::oneVector{Float32}       # (num_v_heads=16,) — log space decay
    ssm_alpha::oneMatrix{Float16}   # (hidden, num_v_heads=16) — dt projection
    ssm_beta::oneMatrix{Float16}    # (hidden, num_v_heads=16) — beta projection
    ssm_conv1d::oneArray{Float32,2} # (conv_kernel=4, conv_channels=6144) — F32
    ssm_dt_bias::oneVector{Float32} # (num_v_heads=16,)
    ssm_norm::RMSNorm               # (head_v_dim=128,) for output norm
    # Conv state buffer (per sequence) — for autoregressive mode
    conv_state::Array{Float32,2}    # CPU buffer: (conv_kernel-1, conv_channels) — ring buffer
    # SSM recurrent state
    ssm_state::Array{Float32,3}     # CPU: (head_v_dim, head_v_dim, num_v_heads) state matrix
    # Dimensions
    num_v_heads::Int    # = 16 (ssm_time_step_rank)
    num_k_heads::Int    # = 16 (ssm_group_count)
    head_k_dim::Int     # = 128 (ssm_state_size)
    head_v_dim::Int     # = 128 (d_inner / num_v_heads)
    d_inner::Int        # = 2048
end

function (m::GatedDeltaNet)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    hidden, seq = size(x)

    # 1. Input projections
    qkv_mixed = mat_mul(m.in_proj, x)    # (6144, seq)
    z = mat_mul(m.gate_proj, x)          # (d_inner=2048, seq)

    # 2. Beta and Alpha projections
    beta_raw = mat_mul(m.ssm_beta, x)    # (num_v_heads=16, seq)
    beta = Float16.(1.0f0 ./ (1.0f0 .+ exp.(Float32.(.-beta_raw))))  # sigmoid

    alpha_raw = mat_mul(m.ssm_alpha, x)  # (num_v_heads=16, seq)
    # Add dt bias and apply softplus
    alpha_32 = Float32.(alpha_raw) .+ reshape(m.ssm_dt_bias, :, 1)
    alpha_sp = Float32.(log.(1.0f0 .+ exp.(alpha_32)))  # softplus

    # Decay gate = alpha_softplus * ssm_a (ssm_a contains -exp(A_log))
    decay_gate = alpha_sp .* reshape(m.ssm_a, :, 1)     # (num_v_heads, seq)

    # 3. Conv1d on qkv_mixed
    # For simplicity in autoregressive mode (seq=1), apply depthwise conv
    # by maintaining a shift buffer
    qkv_cpu = collect(Float32.(qkv_mixed))  # (6144, seq) -> CPU
    conv_channels = size(m.conv_state, 2)
    conv_k = size(m.ssm_conv1d, 1)  # kernel size = 4
    conv_w = collect(Float32.(m.ssm_conv1d))  # (4, 6144) on CPU

    conv_out_cpu = zeros(Float32, conv_channels, seq)

    for t in 1:seq
        # Shift conv state and add new input
        m.conv_state[1:end-1, :] .= @view m.conv_state[2:end, :]
        m.conv_state[end, :] .= @view qkv_cpu[:, t]

        # Depthwise conv: for each channel, dot product of kernel with state
        for c in 1:conv_channels
            s = 0.0f0
            for k in 1:conv_k
                s += conv_w[k, c] * m.conv_state[k, c]
            end
            conv_out_cpu[c, t] = s
        end
    end

    # SiLU activation on conv output
    conv_silu = conv_out_cpu ./ (1.0f0 .+ exp.(-conv_out_cpu))

    # 4. Split into Q, K, V
    # conv_channels = d_inner + 2*num_k_heads*head_k_dim = 2048 + 2*16*128 = 6144
    # Q: (head_k_dim, num_k_heads) = first 2048 elements
    # K: (head_k_dim, num_k_heads) = next 2048 elements
    # V: (head_v_dim, num_v_heads) = last 2048 elements
    qkv_size = m.head_k_dim * m.num_k_heads  # 128*16 = 2048
    v_size = m.head_v_dim * m.num_v_heads     # 128*16 = 2048

    q_flat = @view conv_silu[1:qkv_size, :]                          # (2048, seq)
    k_flat = @view conv_silu[qkv_size+1:2*qkv_size, :]              # (2048, seq)
    v_flat = @view conv_silu[2*qkv_size+1:2*qkv_size+v_size, :]     # (2048, seq)

    # 5. L2-normalize Q and K (per head)
    q_4d = reshape(q_flat, m.head_k_dim, m.num_k_heads, seq)
    k_4d = reshape(k_flat, m.head_k_dim, m.num_k_heads, seq)
    v_4d = reshape(v_flat, m.head_v_dim, m.num_v_heads, seq)

    # L2 norm per head
    for t in 1:seq
        for h in 1:m.num_k_heads
            q_vec = @view q_4d[:, h, t]
            k_vec = @view k_4d[:, h, t]
            q_norm = sqrt(sum(q_vec .^ 2) + m.ssm_norm.eps)
            k_norm_val = sqrt(sum(k_vec .^ 2) + m.ssm_norm.eps)
            q_4d[:, h, t] .= q_vec ./ q_norm
            k_4d[:, h, t] .= k_vec ./ k_norm_val
        end
    end

    # 6. Delta-net recurrence (autoregressive)
    # state: (head_v_dim, head_k_dim, num_v_heads) — outer product accumulation
    # For each timestep t:
    #   decay = exp(gate[v_head, t])
    #   state[v_head] = decay * state[v_head] + beta[v_head,t] * outer(k[k_head,t], v[v_head,t])
    #   output[v_head, t] = q[k_head,t] . state[v_head]
    # num_k_heads == num_v_heads in qwen35 0.8B, so we pair them 1:1

    output_cpu = zeros(Float32, m.head_v_dim, m.num_v_heads, seq)
    decay_cpu = collect(Float32.(decay_gate))  # (num_v_heads, seq)
    beta_cpu = collect(Float32.(beta))          # (num_v_heads, seq)

    for t in 1:seq
        for vh in 1:m.num_v_heads
            # Decay the state
            g = exp(decay_cpu[vh, t])
            b = beta_cpu[vh, t]

            kh = vh  # 1:1 mapping when num_k_heads == num_v_heads
            k_vec = @view k_4d[:, kh, t]  # (head_k_dim,)
            v_vec = @view v_4d[:, vh, t]  # (head_v_dim,)
            q_vec = @view q_4d[:, kh, t]  # (head_k_dim,)

            state = @view m.ssm_state[:, :, vh]  # (head_v_dim, head_k_dim)

            # state = gate * state + beta * outer(v, k)
            state .= g .* state
            for i in 1:m.head_v_dim
                for j in 1:m.head_k_dim
                    state[i, j] += b * v_vec[i] * k_vec[j]
                end
            end

            # output = state @ q
            for i in 1:m.head_v_dim
                s = 0.0f0
                for j in 1:m.head_k_dim
                    s += state[i, j] * q_vec[j]
                end
                output_cpu[i, vh, t] = s
            end
        end
    end

    # 7. Gated normalization: rms_norm(output, ssm_norm) * silu(z)
    # output: (head_v_dim, num_v_heads, seq) -> reshape to (head_v_dim*num_v_heads, seq) = (2048, seq)
    output_flat = reshape(output_cpu, m.head_v_dim * m.num_v_heads, seq)
    output_gpu = oneArray(Float16.(output_flat))

    # RMS norm on chunks of head_v_dim
    # ssm_norm weight is (head_v_dim=128,), applied per-head
    output_normed = similar(output_gpu)
    for vh in 1:m.num_v_heads
        start_idx = (vh-1) * m.head_v_dim + 1
        end_idx = vh * m.head_v_dim
        chunk = @view output_gpu[start_idx:end_idx, :]
        normed = m.ssm_norm(chunk)
        output_normed[start_idx:end_idx, :] .= normed
    end

    # silu(z) gating
    z_silu = silu_f16(z)  # (d_inner=2048, seq)
    gated = output_normed .* z_silu

    # 8. Output projection
    return mat_mul(m.ssm_out, gated)
end

# --- Decoder Layer ---
struct DecoderLayer
    in_norm::RMSNorm
    op::Union{GatedDeltaNet, FullAttention}
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
    x = model.embed[:, tokens]
    for (i, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, caches[i])
    end
    x = model.final_norm(x)
    logits = mat_mul(model.lm_head, x)
    return Float32.(logits) |> collect
end

end # module
