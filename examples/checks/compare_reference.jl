#!/usr/bin/env julia
# compare_reference.jl
#
# Loads the Python reference hidden states and compares them layer-by-layer
# with a CPU Float32 forward pass (avoids GPU Float16 overflow).
#
# Usage:  julia --project=. examples/compare_reference.jl
#
# Prerequisites:  python3 examples/forward_ref.py

using LinearAlgebra, Printf, Statistics

# ── Minimal .npy reader ─────────────────────────────────────────────────

const REFERENCE_DIR = joinpath(@__DIR__, "reference_outputs")
const TOKEN_ID = 151646
const TOKEN_JULIA = TOKEN_ID + 1

function read_npy(path::String)
    open(path, "r") do io
        magic = read(io, 6)
        @assert magic == UInt8[0x93, 'N', 'U', 'M', 'P', 'Y']
        major = read(io, UInt8); read(io, UInt8)
        header_len = major == 1 ? Int(read(io, UInt16)) : Int(read(io, UInt32))
        header_str = String(read(io, header_len))
        dtype_str = match(r"'descr':\s*'([^']+)'", header_str).captures[1]
        shape_str = match(r"'shape':\s*\(([^)]*)\)", header_str).captures[1]
        parts = filter(!isempty, strip.(split(shape_str, ',')))
        shape = isempty(parts) ? () : Tuple(parse.(Int, parts))
        tc = dtype_str[2]; ts = parse(Int, dtype_str[3:end])
        T = tc == 'f' ? (ts == 4 ? Float32 : Float64) :
            tc == 'i' ? (ts == 4 ? Int32 : Int64) : error("bad dtype")
        shape == () && return read(io, T)
        data = Array{T}(undef, shape...)
        read!(io, data)
        return data
    end
end

function load_all_references()
    refs = Dict{String, Array}()
    for f in readdir(REFERENCE_DIR)
        endswith(f, ".npy") || continue
        refs[replace(f, ".npy" => "")] = read_npy(joinpath(REFERENCE_DIR, f))
    end
    return refs
end

function compare_vectors(julia_vec, ref_vec, label::String)
    j = Float64.(vec(julia_vec))
    r = Float64.(vec(ref_vec))
    @assert length(j) == length(r) "Shape mismatch for $label"
    diff = j .- r
    max_err = maximum(abs.(diff))
    rel_err = norm(diff) / (norm(r) + 1e-10)
    cos_sim = dot(j, r) / (norm(j) * norm(r) + 1e-10)
    fmt = Printf.Format("%.4e")
    println("  $label")
    println("    max_err  = $(Printf.format(fmt, max_err))")
    println("    rel_err  = $(Printf.format(fmt, rel_err))")
    println("    cos_sim  = $(Printf.format(Printf.Format("%.8f"), cos_sim))")
end

# ── Load references ─────────────────────────────────────────────────────

println("Loading references from $REFERENCE_DIR...")
refs = load_all_references()
println("  Loaded $(length(refs)) arrays")

# ── Load GGUF weights directly ──────────────────────────────────────────

println("Loading GGUF weights...")
using Inferno
using .Inferno.Loader: extract_tensor, extract_sorted_blocks, get_bias_or_norm
using .Inferno: Model

file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

cfg = let md = file.metadata
    arch_str = get(md, "general.architecture", "qwen35")
    Model.QwenConfig(
        architecture=Symbol(arch_str),
        vocab_size=Int(get(md, "$(arch_str).vocab_size", 248320)),
        hidden_size=Int(md["$(arch_str).embedding_length"]),
        intermediate_size=Int(md["$(arch_str).feed_forward_length"]),
        num_hidden_layers=Int(md["$(arch_str).block_count"]),
        num_attention_heads=Int(md["$(arch_str).attention.head_count"]),
        num_key_value_heads=Int(md["$(arch_str).attention.head_count_kv"]),
        head_dim=Int(md["$(arch_str).attention.key_length"]),
        rms_norm_eps=Float32(md["$(arch_str).attention.layer_norm_rms_epsilon"]),
        rope_theta=Float32(md["$(arch_str).rope.freq_base"]),
        max_position_embeddings=min(4096, Int(md["$(arch_str).context_length"])),
        full_attention_interval=Int(md["$(arch_str).full_attention_interval"]),
        ssm_inner_size=Int(md["$(arch_str).ssm.inner_size"]),
        ssm_state_size=Int(md["$(arch_str).ssm.state_size"]),
        ssm_group_count=Int(md["$(arch_str).ssm.group_count"]),
        ssm_time_step_rank=Int(md["$(arch_str).ssm.time_step_rank"]),
        ssm_conv_kernel=Int(md["$(arch_str).ssm.conv_kernel"]),
        num_experts=0, num_experts_per_tok=0,
        q_lora_rank=0, kv_lora_rank=0, qk_rope_head_dim=0, v_head_dim=0,
    )
end

# Helper: load weight as CPU Float32 matrix, transposed to match get_weight convention
function load_mat(name)
    t = extract_tensor(file, name)
    if ndims(t) == 1
        return Float32.(vec(collect(t)))
    else
        return Float32.(collect(t'))
    end
end

function load_vec(name)
    return Float32.(vec(collect(extract_tensor(file, name))))
end

# Load all weights
embed_mat = Float32.(collect(extract_tensor(file, "token_embd.weight")))  # (hidden, vocab)
output_norm_w = load_vec("output_norm.weight")
lm_head = haskey(file.tensors, "output.weight") ? load_mat("output.weight") : embed_mat'

blocks_raw = extract_sorted_blocks(file.tensors)

struct BlockWeights
    is_ssm::Bool
    attn_norm::Vector{Float32}
    post_attn_norm::Vector{Float32}
    ffn_gate::Matrix{Float32}
    ffn_up::Matrix{Float32}
    ffn_down::Matrix{Float32}
    # SSM
    attn_qkv::Matrix{Float32}
    attn_gate::Matrix{Float32}
    ssm_out::Matrix{Float32}
    conv1d::Matrix{Float32}
    alpha::Matrix{Float32}
    beta::Matrix{Float32}
    ssm_a::Vector{Float32}
    dt_bias::Vector{Float32}
    ssm_norm::Vector{Float32}
    # Attention
    attn_q::Matrix{Float32}
    attn_k::Matrix{Float32}
    attn_v::Matrix{Float32}
    attn_output::Matrix{Float32}
    attn_q_norm::Vector{Float32}
    attn_k_norm::Vector{Float32}
end

function load_block(blk)
    is_ssm = blk.attn_qkv_weight !== nothing
    function safe_mat(f)
        f === nothing ? zeros(Float32, 1, 1) : load_mat(f.name)
    end
    function safe_vec(f)
        f === nothing ? Float32[] : load_vec(f.name)
    end
    BlockWeights(
        is_ssm,
        safe_vec(blk.attn_norm_weight),
        safe_vec(blk.post_attention_norm_weight),
        safe_mat(blk.ffn_gate_weight),
        safe_mat(blk.ffn_up_weight),
        safe_mat(blk.ffn_down_weight),
        safe_mat(blk.attn_qkv_weight),
        safe_mat(blk.attn_gate_weight),
        safe_mat(blk.ssm_out_weight),
        safe_mat(blk.ssm_conv1d_weight),
        safe_mat(blk.ssm_alpha_weight),
        safe_mat(blk.ssm_beta_weight),
        safe_vec(blk.ssm_a),
        safe_vec(blk.ssm_dt_bias),
        safe_vec(blk.ssm_norm_weight),
        safe_mat(blk.attn_q_weight),
        safe_mat(blk.attn_k_weight),
        safe_mat(blk.attn_v_weight),
        safe_mat(blk.attn_output_weight),
        safe_vec(blk.attn_q_norm_weight),
        safe_vec(blk.attn_k_norm_weight),
    )
end

block_weights = [load_block(b) for b in blocks_raw]

# ── CPU Float32 operations ──────────────────────────────────────────────

function cpu_rmsnorm(x, w, eps)
    ss = sum(Float64.(x) .^ 2)
    scale = Float32(1.0 / sqrt(ss / length(x) + eps))
    return x .* scale .* w
end

function cpu_silu(x)
    return x .* Float32.(1.0 ./ (1.0 .+ exp.(-Float64.(x))))
end

# ── CPU forward pass ────────────────────────────────────────────────────

function cpu_ssm_forward(bw, x_norm, conv_state, h_state, cfg)
    inner = cfg.ssm_inner_size
    state_size = cfg.ssm_state_size
    groups = cfg.ssm_group_count
    head_v_dim = inner ÷ cfg.ssm_time_step_rank
    conv_channels = 2 * groups * state_size + inner
    K = cfg.ssm_conv_kernel
    eps = Float32(cfg.rms_norm_eps)

    qkv = bw.attn_qkv * x_norm
    z_buf = bw.attn_gate * x_norm

    # Conv1d
    if K > 1
        conv_state[:, 1:K-1] = conv_state[:, 2:K]
    end
    conv_state[:, K] = qkv
    x_conv = zeros(Float32, conv_channels)
    for k in 1:K
        x_conv .+= conv_state[:, k] .* bw.conv1d[:, k]
    end
    x_conv = cpu_silu(x_conv)

    # Split
    qk_size = state_size * groups
    q_all = reshape(x_conv[1:qk_size], state_size, groups)
    k_all = reshape(x_conv[qk_size+1:2*qk_size], state_size, groups)
    v_all = reshape(x_conv[2*qk_size+1:2*qk_size+inner], head_v_dim, groups)

    # Projections
    x64 = Float64.(x_norm)
    alpha_proj = Float64.(bw.alpha) * x64
    beta_proj = Float64.(bw.beta) * x64
    ssm_a = Float64.(bw.ssm_a)
    dt_bias = Float64.(bw.dt_bias)

    y_all = zeros(Float64, inner)
    for g in 1:groups
        qg = Float64.(q_all[:, g])
        kg = Float64.(k_all[:, g])
        vg = Float64.(v_all[:, g])

        qn = sqrt(sum(qg .^ 2) + Float64(eps))
        kn = sqrt(sum(kg .^ 2) + Float64(eps))
        q_norm = qg ./ qn
        k_norm = kg ./ kn

        dg = exp(log(1.0 + exp(alpha_proj[g] + dt_bias[g])) * ssm_a[g])
        bg = 1.0 / (1.0 + exp(-beta_proj[g]))

        h_state[:, :, g] .*= dg
        sk = h_state[:, :, g] * k_norm
        update = bg .* (vg .- sk)
        h_state[:, :, g] .+= update * k_norm'
        yg = h_state[:, :, g] * q_norm
        y_all[(g-1)*head_v_dim+1:g*head_v_dim] .= yg
    end

    # Output norm + gate
    ssm_norm_w = Float64.(bw.ssm_norm)
    y_reshaped = reshape(y_all, head_v_dim, groups)
    y_normed = zeros(Float64, inner)
    for g in 1:groups
        v = y_reshaped[:, g]
        ss = sum(v .^ 2)
        scale = 1.0 / sqrt(ss / head_v_dim + Float64(eps))
        y_normed[(g-1)*head_v_dim+1:g*head_v_dim] .= v .* scale .* ssm_norm_w
    end
    z_gated = Float64.(cpu_silu(z_buf))
    y_normed .*= z_gated
    return Float32.(Float64.(bw.ssm_out) * y_normed)
end

function cpu_attn_forward(bw, x_norm, kv_cache_k, kv_cache_v, pos, cfg, sin_cache, cos_cache)
    head_dim = cfg.head_dim
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    rope_dim = head_dim  # qk_rope_head_dim = 0
    eps = Float32(cfg.rms_norm_eps)

    q_all = bw.attn_q * x_norm
    k_buf = bw.attn_k * x_norm
    v_buf = bw.attn_v * x_norm

    q_size = nq * head_dim
    q = q_all[1:q_size]
    gate = q_all[q_size+1:2*q_size]
    q_heads = reshape(q, head_dim, nq)
    k_heads = reshape(k_buf, head_dim, nkv)
    v_heads = reshape(v_buf, head_dim, nkv)

    # Q/K norm
    for h in 1:nq
        q_heads[:, h] = cpu_rmsnorm(q_heads[:, h], bw.attn_q_norm, eps)
    end
    for h in 1:nkv
        k_heads[:, h] = cpu_rmsnorm(k_heads[:, h], bw.attn_k_norm, eps)
    end

    # SiLU gate * Q
    gate_gated = cpu_silu(reshape(gate, head_dim, nq))
    q_heads .*= gate_gated

    # RoPE
    rope_pairs = rope_dim ÷ 2
    for h in 1:nq
        for p in 1:rope_pairs
            i0 = 2p - 1; i1 = 2p
            s = Float64(sin_cache[p, pos])
            c = Float64(cos_cache[p, pos])
            q0 = Float64(q_heads[i0, h])
            q1 = Float64(q_heads[i1, h])
            q_heads[i0, h] = Float32(q0 * c - q1 * s)
            q_heads[i1, h] = Float32(q0 * s + q1 * c)
        end
    end
    for h in 1:nkv
        for p in 1:rope_pairs
            i0 = 2p - 1; i1 = 2p
            s = Float64(sin_cache[p, pos])
            c = Float64(cos_cache[p, pos])
            k0 = Float64(k_heads[i0, h])
            k1 = Float64(k_heads[i1, h])
            k_heads[i0, h] = Float32(k0 * c - k1 * s)
            k_heads[i1, h] = Float32(k0 * s + k1 * c)
        end
    end

    # Update KV cache
    kv_cache_k[:, :, pos] = k_heads
    kv_cache_v[:, :, pos] = v_heads

    # Attention
    scale = Float64(1.0 / sqrt(head_dim))
    attn_out = zeros(Float64, head_dim, nq)
    gqa_ratio = nq ÷ nkv
    for h in 1:nq
        kv_h = (h - 1) ÷ gqa_ratio + 1
        K_past = Float64.(kv_cache_k[:, kv_h, 1:pos])
        V_past = Float64.(kv_cache_v[:, kv_h, 1:pos])
        q_h = Float64.(q_heads[:, h])
        scores = K_past' * q_h
        scores .*= scale
        scores .-= maximum(scores)
        scores = exp.(scores)
        scores ./= sum(scores)
        attn_out[:, h] = V_past * scores
    end

    attn_out_flat = vec(attn_out)
    return Float32.(Float64.(bw.attn_output) * attn_out_flat)
end

function cpu_mlp_forward(bw, x_norm)
    gate = cpu_silu(bw.ffn_gate * x_norm)
    up = bw.ffn_up * x_norm
    gate .*= up
    return bw.ffn_down * gate
end

# ── Run comparison ──────────────────────────────────────────────────────

function run_comparison()
    println("\nRunning CPU Float32 forward pass (token $TOKEN_ID)...\n")

    # Embedding
    x = Float32.(embed_mat[:, TOKEN_JULIA])
    compare_vectors(x, refs["embed"], "Embedding")

    hidden = cfg.hidden_size
    state_size = cfg.ssm_state_size
    groups = cfg.ssm_group_count
    inner = cfg.ssm_inner_size
    head_v_dim = inner ÷ cfg.ssm_time_step_rank
    conv_channels = 2 * groups * state_size + inner
    K = cfg.ssm_conv_kernel
    head_dim = cfg.head_dim
    nkv = cfg.num_key_value_heads
    max_pos = cfg.max_position_embeddings
    eps = Float32(cfg.rms_norm_eps)

    # Initialize SSM states
    conv_states = [zeros(Float32, conv_channels, K) for _ in 1:cfg.num_hidden_layers]
    h_states = [zeros(Float64, head_v_dim, state_size, groups) for _ in 1:cfg.num_hidden_layers]

    # Initialize KV caches
    kv_k = [zeros(Float32, head_dim, nkv, max_pos) for _ in 1:cfg.num_hidden_layers]
    kv_v = [zeros(Float32, head_dim, nkv, max_pos) for _ in 1:cfg.num_hidden_layers]

    # Build RoPE cache
    rope_dim = head_dim
    rope_pairs = rope_dim ÷ 2
    rope_theta = Float64(cfg.rope_theta)
    sin_cache = zeros(Float32, rope_pairs, max_pos)
    cos_cache = zeros(Float32, rope_pairs, max_pos)
    for p in 1:rope_pairs
        i = 2p - 1
        freq = 1.0 / (rope_theta ^ ((i - 1) / rope_dim))
        for pos_idx in 1:max_pos
            sin_cache[p, pos_idx] = Float32(sin((pos_idx - 1) * freq))
            cos_cache[p, pos_idx] = Float32(cos((pos_idx - 1) * freq))
        end
    end

    pos = 1

    for i in 1:cfg.num_hidden_layers
        bw = block_weights[i]

        # Branch 1
        x_norm1 = cpu_rmsnorm(x, bw.attn_norm, eps)

        if bw.is_ssm
            branch_out = cpu_ssm_forward(bw, x_norm1, conv_states[i], h_states[i], cfg)
            label = "SSM"
        else
            branch_out = cpu_attn_forward(bw, x_norm1, kv_k[i], kv_v[i], pos, cfg, sin_cache, cos_cache)
            label = "ATN"
        end

        x = x .+ branch_out
        haskey(refs, "layer$(i-1)_post_branch") &&
            compare_vectors(x, refs["layer$(i-1)_post_branch"], "Layer $(i-1) ($label) post-branch")

        # Branch 2: MLP
        x_norm2 = cpu_rmsnorm(x, bw.post_attn_norm, eps)
        mlp_out = cpu_mlp_forward(bw, x_norm2)
        x = x .+ mlp_out

        haskey(refs, "layer$(i-1)_post_mlp") &&
            compare_vectors(x, refs["layer$(i-1)_post_mlp"], "Layer $(i-1) ($label) post-mlp")

        println()
    end

    # Final norm
    hidden_final = cpu_rmsnorm(x, output_norm_w, eps)
    compare_vectors(hidden_final, refs["final_norm"], "Final norm output")

    # Logits
    logits = Float64.(lm_head) * Float64.(hidden_final)
    compare_vectors(logits, refs["logits"], "Logits")

    top5_julia = sortperm(logits, rev=true)[1:5]
    top5_ref = sortperm(Float64.(vec(refs["logits"])), rev=true)[1:5]
    println("\n  Julia top-5: $top5_julia")
    println("  Ref   top-5: $top5_ref")
    println("\nDone.")
end

run_comparison()
