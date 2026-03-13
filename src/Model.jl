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
    weight::oneVector{Float32}
    eps::Float32
end

function (norm::RMSNorm)(x::oneAPI.oneArray{Float32, N}) where N
    # GPU-native RMSNorm using broadcast to avoid transfers
    x2 = x .* x
    m = mean(x2, dims=1)
    inv_rms = 1.0f0 ./ sqrt.(m .+ norm.eps)
    return x .* inv_rms .* norm.weight
end

function (norm::RMSNorm)(x::AbstractArray{Float32, N}) where N
    # CPU fallback for non-GPU arrays
    x2 = x .* x
    m = mean(x2, dims=1)
    inv_rms = 1.0f0 ./ sqrt.(m .+ norm.eps)
    return x .* inv_rms .* collect(norm.weight)
end

# --- Matrix Multiplication (Fallback using broadcast to bypass crashes) ---
# Custom kernel for stable matrix-vector multiplication
# Avoids oneMKL/GEMM driver issues
function mat_mul_kernel!(res, weight, x, K, N)
    n = get_global_id(1)
    if n <= N
        val = 0.0f0
        for k in 1:K
            @inbounds val += weight[n, k] * x[k, 1]
        end
        res[n] = val
    end
    return nothing
end

function mat_mul(weight::AbstractArray{Float32,2}, x::AbstractArray{Float32,2})
    # weight: (N, K), x: (K, S) -> result: (N, S)
    return Float32.(collect(weight) * collect(x))
end

function mat_mul(weight::oneMatrix{Float32}, x::oneMatrix{Float32})
    N, K = size(weight)
    S = size(x, 2)

    if S == 1
        res = oneArray{Float32}(undef, N)
        gs = min(N, 256)
        gr = cld(N, gs)
        @oneapi items=gs groups=gr mat_mul_kernel!(res, weight, x, K, N)
        return reshape(res, N, 1)
    else
        res = oneArray(zeros(Float32, N, S))
        gs = min(N, 256)
        gr = cld(N, gs)
        for s in 1:S
            v = @view x[:, s]
            r = @view res[:, s]
            @oneapi items=gs groups=gr mat_mul_kernel!(r, weight, v, K, N)
        end
        return res
    end
end

function mat_mul_AB_kernel!(res, A, B, N, M, S_dim)
    m = get_global_id(1)
    s = get_global_id(2)
    if m <= M && s <= S_dim
        val = 0.0f0
        for n in 1:N
            @inbounds val += A[m, n] * B[n, s]
        end
        res[m, s] = val
    end
    return nothing
end

function mat_mul_AB(A::AbstractArray{Float32,2}, B::AbstractArray{Float32,2})
    # A: (M, N), B: (N, S) -> result: (M, S)
    return Float32.(collect(A) * collect(B))
end

function mat_mul_AB(A::oneMatrix{Float32}, B::oneMatrix{Float32})
    M, N = size(A)
    S = size(B, 2)
    res = oneArray{Float32}(undef, M, S)
    gs_x = min(M, 16)
    gs_y = min(S, 16)
    gr_x = cld(M, gs_x)
    gr_y = cld(S, gs_y)
    @oneapi items=(gs_x, gs_y) groups=(gr_x, gr_y) mat_mul_AB_kernel!(res, A, B, N, M, S)
    return res
end# --- SiLU ---
function silu(x::AbstractArray{Float32})
    return x .* (1.0f0 ./ (1.0f0 .+ exp.(-x)))
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

function (rope::RotaryEmbedding)(x::oneArray{Float32,3}, pos::Int)
    d, h, seq = size(x)
    d_rope = min(d, rope.dim)
    positions = oneArray(Float32.(pos:(pos+seq-1)))
    freqs = rope.inv_freq[1:(d_rope÷2)] * positions'
    cos_t = reshape(cos.(freqs), d_rope ÷ 2, 1, seq)
    sin_t = reshape(sin.(freqs), d_rope ÷ 2, 1, seq)
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
    k::oneArray{Float32,3}  # (head_dim, n_kv, max_seq)
    v::oneArray{Float32,3}
    pos::Int
end

function init_kv_cache(head_dim, n_kv, max_seq)
    k = zeros(Float32, head_dim, n_kv, max_seq) |> oneArray
    v = zeros(Float32, head_dim, n_kv, max_seq) |> oneArray
    return KVCache(k, v, 0)
end

function update_kv_cache!(cache::KVCache, k::oneArray{Float32,3}, v::oneArray{Float32,3})
    seq = size(k, 3)
    pos = cache.pos
    dk = size(cache.k, 1)
    nk = size(cache.k, 2)
    @views cache.k[1:dk, 1:nk, pos+1:pos+seq] .= k
    @views cache.v[1:dk, 1:nk, pos+1:pos+seq] .= v
    cache.pos += seq
    return cache.k, cache.v
end

# --- MLP ---
struct MLP
    gate_weight::oneMatrix{Float32}
    up_weight::oneMatrix{Float32}
    down_weight::oneMatrix{Float32}
end

function (m::MLP)(x::oneMatrix{Float32})
    # silu(gate) * up
    g = mat_mul(m.gate_weight, x)
    u = mat_mul(m.up_weight, x)

    # SiLU: x * sigmoid(x)
    @. g = g * (1.0f0 / (1.0f0 + exp(-g)))

    res = g .* u
    return mat_mul(m.down_weight, res)
end

# --- Full Attention Layer ---
# Q weight packs both Q and gate interleaved: attn_q output is (head_dim*2*n_heads, seq)
# Q = output[0:head_dim, h, :], gate = output[head_dim:2*head_dim, h, :]
struct FullAttention
    wq::oneMatrix{Float32}
    wk::oneMatrix{Float32}
    wv::oneMatrix{Float32}
    wo::oneMatrix{Float32}
    q_norm::RMSNorm
    k_norm::RMSNorm
    n_heads::Int
    n_kv::Int
    head_dim::Int
end

function (m::FullAttention)(x::oneArray{Float32,2}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    hd, seq = m.head_dim, size(x, 2)

    # 1. Packed Q+gate projection
    q_full = mat_mul(m.wq, x)  # (hd*2*n_heads, seq)

    # Split into Q and gate
    q_3d = reshape(q_full, hd * 2, m.n_heads, seq)
    q_only = q_3d[1:hd, :, :]
    gate_raw = q_3d[hd+1:2*hd, :, :]

    # 2. K, V projections
    k = mat_mul(m.wk, x)  # (hd*n_kv, seq)
    v = mat_mul(m.wv, x)  # (hd*n_kv, seq)

    # 3. Apply Q, K normalization
    q_normed = m.q_norm(reshape(q_only, hd, :))
    k_normed = m.k_norm(reshape(k, hd, :))

    q_2d = reshape(q_normed, hd, m.n_heads, seq)
    k_2d = reshape(k_normed, hd, m.n_kv, seq)
    v_2d = reshape(v, hd, m.n_kv, seq)

    # 4. RoPE
    q_rope = rope(q_2d, pos)
    k_rope = rope(k_2d, pos)

    # 5. Gating Q
    gate_silu = gate_raw .* (1.0f0 ./ (1.0f0 .+ exp.(-gate_raw)))
    q_gated = q_rope .* gate_silu

    # 6. KV Cache
    K, V = update_kv_cache!(cache, k_rope, v_2d)

    # 7. Attention
    total_len = cache.pos
    scale = 1.0f0 / sqrt(Float32(hd))

    # Combined output buffer
    if seq == 1
        q_final = reshape(q_gated, hd, m.n_heads, 1) # (hd, n_heads, 1)
        kv_per_q = m.n_heads ÷ m.n_kv
        outputs = []
        for h in 1:m.n_heads
            kh = (h - 1) ÷ kv_per_q + 1
            # scores: (total_len, 1)
            scores = mat_mul(K[:, kh, 1:total_len], q_final[:, h, :]) .* scale

            s32 = collect(scores)
            mx = maximum(s32)
            ex = exp.(s32 .- mx)
            pb = oneArray(reshape(ex ./ sum(ex), :, 1)) # (total_len, 1)

            # out_h: (hd, 1)
            out_h = mat_mul_AB(V[:, kh, 1:total_len], pb)
            push!(outputs, out_h)
        end
        combined = reduce(vcat, outputs)
    else
        q_final = reshape(q_gated, hd, m.n_heads, seq)
        kv_per_q = m.n_heads ÷ m.n_kv
        combined_all = zeros(Float32, hd * m.n_heads, seq) |> oneArray
        for h in 1:m.n_heads
            kh = (h - 1) ÷ kv_per_q + 1
            for s in 1:seq
                # scores: (pos+s, 1)
                scores = mat_mul(K[:, kh, 1:(pos+s)], reshape(q_final[:, h, s], :, 1)) .* scale
                s32 = collect(scores)
                mx = maximum(s32)
                ex = exp.(s32 .- mx)
                pb = oneArray(reshape(ex ./ sum(ex), :, 1))
                # out_s: (hd, 1)
                out_s = mat_mul_AB(V[:, kh, 1:(pos+s)], pb)
                combined_all[(h-1)*hd+1:h*hd, s] .= vec(out_s)
            end
        end
        combined = combined_all
    end

    # 8. Output projection
    return mat_mul(m.wo, combined)
end

# --- Gated Delta Net (SSM Layer) ---
# Reference: qwen35.cpp build_layer_attn_linear
struct GatedDeltaNet
    in_proj::oneMatrix{Float32}     # wqkv: (hidden, 6144) — projects to qkv_mixed
    gate_proj::oneMatrix{Float32}   # wqkv_gate: (hidden, d_inner=2048) — projects to z
    ssm_out::oneMatrix{Float32}     # (d_inner, hidden)
    ssm_a::oneVector{Float32}       # (num_v_heads=16,) — log space decay
    ssm_alpha::oneMatrix{Float32}   # (hidden, num_v_heads=16) — dt projection
    ssm_beta::oneMatrix{Float32}    # (hidden, num_v_heads=16) — beta projection
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
    # Pre-collected CPU weights for efficiency
    ssm_conv1d_cpu::Array{Float32,2}
end

function reset_states!(m::GatedDeltaNet)
    m.conv_state .= 0
    m.ssm_state .= 0
end

function (m::GatedDeltaNet)(x::oneMatrix{Float32}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    hidden, seq = size(x)

    # 1. Input projections
    qkv_mixed = mat_mul(m.in_proj, x)    # (6144, seq)
    z = mat_mul(m.gate_proj, x)          # (d_inner=2048, seq)
    oneAPI.synchronize()
    # 2. Beta and Alpha projections
    beta_raw = mat_mul(m.ssm_beta, x)    # (num_v_heads=16, seq)
    alpha_raw = mat_mul(m.ssm_alpha, x)  # (num_v_heads=16, seq)
    oneAPI.synchronize()

    # Move raw projections to CPU to avoid unstable GPU broadcast kernels
    beta_cpu_raw = collect(beta_raw)
    alpha_cpu_raw = collect(alpha_raw)

    # Compute Beta and Alpha on CPU
    beta = 1.0f0 ./ (1.0f0 .+ exp.(-Float32.(beta_cpu_raw))) # sigmoid
    alpha_sp = log.(1.0f0 .+ exp.(Float32.(alpha_cpu_raw) .+ collect(m.ssm_dt_bias))) # softplus

    # Decay gate = alpha_softplus * ssm_a
    # ssm_a is on GPU, let's bring it once
    ssm_a_cpu = collect(m.ssm_a)
    decay_gate = alpha_sp .* ssm_a_cpu     # (num_v_heads, seq)

    # 3. Conv1d on qkv_mixed
    # Move mixed to CPU
    qkv_cpu_h16 = collect(qkv_mixed)
    qkv_cpu = Float32.(qkv_cpu_h16)

    conv_channels = size(m.conv_state, 2)
    conv_k = size(m.ssm_conv1d, 1)  # kernel size = 4
    conv_w = m.ssm_conv1d_cpu

    conv_out_cpu = zeros(Float32, conv_channels, seq)

    for t in 1:seq
        m.conv_state[1:end-1, :] .= @view m.conv_state[2:end, :]
        m.conv_state[end, :] .= @view qkv_cpu[:, t]

        for c in 1:conv_channels
            s = 0.0f0
            for k in 1:conv_k
                s += conv_w[k, c] * m.conv_state[k, c]
            end
            conv_out_cpu[c, t] = s
        end
    end

    # SiLU on CPU
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
            g = exp(decay_cpu[vh, t])
            b = beta_cpu[vh, t]

            kh = vh
            k_vec = @view k_4d[:, kh, t]
            v_vec = @view v_4d[:, vh, t]
            q_vec = @view q_4d[:, kh, t]

            state = @view m.ssm_state[:, :, vh]

            # Optimized state update: state = g * state + b * v * k'
            # (head_v_dim, head_k_dim)
            @inbounds for i in 1:m.head_v_dim
                vi_b = v_vec[i] * b
                @simd for j in 1:m.head_k_dim
                    state[i, j] = g * state[i, j] + vi_b * k_vec[j]
                end
            end

            # Output = state @ q
            @inbounds for i in 1:m.head_v_dim
                s = 0.0f0
                @simd for j in 1:m.head_k_dim
                    s += state[i, j] * q_vec[j]
                end
                output_cpu[i, vh, t] = s
            end
        end
    end

    # 7. Gated normalization: rms_norm(output, ssm_norm) * silu(z)
    # output: (head_v_dim, num_v_heads, seq)
    output_flat = reshape(output_cpu, m.head_v_dim * m.num_v_heads, seq)
    output_gpu = oneArray(Float32.(output_flat))

    # Batched RMS norm across heads
    # Reshape to (head_v_dim, num_v_heads * seq) to use existing RMSNorm logic?
    # No, let's just do it directly here for efficiency
    out_3d = reshape(output_gpu, m.head_v_dim, m.num_v_heads * seq)
    rms = sqrt.(mean(out_3d .* out_3d, dims=1) .+ m.ssm_norm.eps)
    w = reshape(m.ssm_norm.weight, :, 1)
    output_normed_batched = (out_3d ./ rms) .* w
    output_normed = reshape(output_normed_batched, m.head_v_dim * m.num_v_heads, seq)

    # silu(z) gating
    z_silu = silu(z)  # (d_inner=2048, seq)
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

function (layer::DecoderLayer)(x::oneMatrix{Float32}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    h = layer.in_norm(x)
    h = layer.op(h, pos, rope, cache)
    x .+= h
    h = layer.post_norm(x)
    h = layer.mlp(h)
    x .+= h
    return x
end

# --- Model ---
struct QwenModel
    config::QwenConfig
    embed::Matrix{Float32} # CPU-based
    layers::Vector{DecoderLayer}
    final_norm::RMSNorm
    lm_head::Matrix{Float32} # CPU-based (huge vocab)
    rope::RotaryEmbedding
end

function forward!(model::QwenModel, tokens::Vector{Int}, pos::Int, caches::Vector{KVCache})
    try
        # 1. Embedding (CPU to GPU as F32)
        indices = tokens .+ 1
        emb_rows = model.embed[:, indices]
        x = oneArray(Float32.(emb_rows))
        oneAPI.synchronize()

        # 2. Layers
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos, model.rope, caches[i])
            oneAPI.synchronize()
            if i % 24 == 0
                GC.gc(false)
            end
        end

        # 3. Final Norm and Logits
        x_cpu_out = collect(x)
        x_cpu_norm = model.final_norm(x_cpu_out) # Now on CPU

        # model.lm_head: (hidden, vocab)
        # x_cpu_norm: (hidden, seq) — but it was returned as oneArray by RMSNorm
        x_final = collect(x_cpu_norm)

        # We want (vocab, seq) = (vocab, hidden) * (hidden, seq)
        logits = (model.lm_head') * x_final # (vocab, seq)

        return Float32.(logits)
    catch e
        println("ERROR in forward!: ", e)
        st = stacktrace(catch_backtrace())
        for line in st
            println("  ", line)
        end
        rethrow(e)
    end
end

function reset_states!(model::QwenModel)
    for layer in model.layers
        if layer.is_ssm
            reset_states!(layer.op)
        end
    end
end

end # module
