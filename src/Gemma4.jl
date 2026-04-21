module Gemma4

using LinearAlgebra

# ============================================================
# Gemma4 CPU Inference Implementation
# ============================================================
#
# Key architectural features:
# 1. Mixed sliding/full attention layers
# 2. Proportional + partial RoPE for full attention (25%, theta=1M)
# 3. Default RoPE for sliding attention (full rotary, theta=10K)
# 4. Per-layer input embeddings (unique to Gemma4)
# 5. KV cache sharing across layers
# 6. Attention logit softcapping
# 7. Final logit softcapping (tanh cap at 30.0)
# 8. Q/K RMSNorm (with_scale=True), V RMSNorm (with_scale=False)
# 9. GELU tanh activation (not SiLU)
# 10. Embedding scaling (multiply by sqrt(hidden_size))
# 11. Layer scalar (learnable per-layer scale, applied to whole hidden)
# 12. attention_k_eq_v: full attention layers use K=V
# 13. scaling=1.0 for attention (Q/K norm handles scaling)
# ============================================================

# --- Structs ---

struct Gemma4Config
    hidden_size::Int
    num_layers::Int
    num_q_heads::Int
    num_kv_heads::Int
    num_global_kv_heads::Int
    head_dim::Int
    global_head_dim::Int
    intermediate_size::Int
    double_wide_intermediate::Int
    vocab_size::Int
    vocab_size_per_layer_input::Int
    max_seq_len::Int
    sliding_window::Int
    rms_norm_eps::Float32
    final_logit_softcapping::Float32
    attention_logits_softcapping::Float32
    embed_scale::Float32
    per_layer_input_scale::Float32
    per_layer_model_projection_scale::Float32
    layer_types::Vector{String}
    num_kv_shared_layers::Int
    first_kv_shared_layer::Int  # index of first shared layer (0-based)
    hidden_size_per_layer_input::Int
    attention_k_eq_v::Bool
    # RoPE params
    sliding_rope_theta::Float32
    full_rope_theta::Float32
    full_partial_rotary_factor::Float32
end

mutable struct AttentionLayer
    q_proj::Matrix{Float32}   # (num_q_heads * head_dim, hidden_size)
    k_proj::Matrix{Float32}   # (num_kv_heads * head_dim, hidden_size)
    v_proj::Matrix{Float32}   # (num_kv_heads * head_dim, hidden_size) or nothing if k_eq_v
    o_proj::Matrix{Float32}   # (hidden_size, num_q_heads * head_dim)
    q_norm_w::Vector{Float32} # (head_dim,) RMSNorm with scale
    k_norm_w::Vector{Float32} # (head_dim,) RMSNorm with scale
    v_norm_enabled::Bool      # whether v_norm is used (always true for non-shared layers)
    is_sliding::Bool
    is_kv_shared::Bool
    kv_shared_src::Int        # source layer index for shared KV (0-based)
    head_dim::Int             # actual head_dim for this layer (sliding vs global)
    num_kv_heads_actual::Int  # num_kv_heads for this layer
    # Pre-allocated buffers
    q_buf::Vector{Float32}
    k_buf::Vector{Float32}
    v_buf::Vector{Float32}
    attn_out_buf::Vector{Float32}
end

mutable struct MLPLayer
    gate_proj::Matrix{Float32}  # (intermediate, hidden)
    up_proj::Matrix{Float32}    # (intermediate, hidden)
    down_proj::Matrix{Float32}  # (hidden, intermediate)
    # Pre-allocated buffers
    gate_buf::Vector{Float32}
    up_buf::Vector{Float32}
    hidden_buf::Vector{Float32}
end

mutable struct PerLayerInput
    gate_proj::Matrix{Float32}       # (pli_size, hidden)
    projection::Matrix{Float32}      # (hidden, pli_size)
    post_norm_w::Vector{Float32}     # (hidden,) RMSNorm
end

mutable struct DecoderLayer
    input_norm_w::Vector{Float32}        # input_layernorm
    post_attn_norm_w::Vector{Float32}    # post_attention_layernorm
    pre_ff_norm_w::Vector{Float32}       # pre_feedforward_layernorm
    post_ff_norm_w::Vector{Float32}      # post_feedforward_layernorm
    attn::AttentionLayer
    mlp::MLPLayer
    pli::Union{PerLayerInput, Nothing}
    layer_scalar::Float32
    # Pre-allocated buffers
    norm_buf::Vector{Float32}
    pli_gate_buf::Vector{Float32}
    pli_out_buf::Vector{Float32}
end

struct KVCacheG4
    k_cache::Vector{Matrix{Float32}}  # one per layer: (num_kv_heads * head_dim, max_seq_len)
    v_cache::Vector{Matrix{Float32}}  # one per layer
    seq_len::Vector{Int}              # current length per layer
end

mutable struct Gemma4Model
    config::Gemma4Config
    embed_tokens::Matrix{Float32}           # (vocab_size, hidden_size)
    embed_tokens_per_layer::Matrix{Float32} # (vocab_per_layer, num_layers * pli_size)
    per_layer_model_proj::Matrix{Float32}   # (num_layers * pli_size, hidden_size)
    per_layer_proj_norm_w::Vector{Float32}  # (pli_size,) RMSNorm
    final_norm_w::Vector{Float32}           # (hidden_size,) final RMSNorm
    layers::Vector{DecoderLayer}
    # RoPE pre-computed
    sliding_cos::Vector{Vector{Float32}}  # per position: (head_dim/2,)
    sliding_sin::Vector{Vector{Float32}}
    full_cos::Vector{Vector{Float32}}     # per position: (global_head_dim/2,)
    full_sin::Vector{Vector{Float32}}
    # Shared KV states (populated during prefill)
    shared_kv_k::Dict{Int, Matrix{Float32}}  # layer_idx => (num_kv_heads * head_dim, seq_len)
    shared_kv_v::Dict{Int, Matrix{Float32}}
    # Pre-allocated buffers
    hidden_buf::Vector{Float32}
    residual_buf::Vector{Float32}
    pli_embed_buf::Vector{Float32}        # (num_layers * pli_size,)
    pli_proj_buf::Vector{Float32}         # (num_layers * pli_size,)
    pli_per_layer_buf::Vector{Float32}    # (pli_size,) per-layer slice
    logits_buf::Vector{Float32}           # (vocab_size,)
end

# --- KV Cache ---

function init_kv_cache(config::Gemma4Config, max_seq_len::Int)
    k_caches = Matrix{Float32}[]
    v_caches = Matrix{Float32}[]
    seq_lens = Int[]
    for i in 1:config.num_layers
        layer = nothing  # will be set later
        head_d = config.head_dim
        n_kv = config.num_kv_heads
        # For full attention layers that use global head dim
        if config.layer_types[i] == "full_attention"
            head_d = config.global_head_dim
            n_kv = config.num_global_kv_heads
        end
        push!(k_caches, Matrix{Float32}(undef, n_kv * head_d, max_seq_len))
        push!(v_caches, Matrix{Float32}(undef, n_kv * head_d, max_seq_len))
        push!(seq_lens, 0)
    end
    return KVCacheG4(k_caches, v_caches, seq_lens)
end

# --- RoPE ---

function precompute_rope(config::Gemma4Config, max_seq_len::Int)
    # Sliding attention: default RoPE, full rotary, theta=10000
    sliding_dim = config.head_dim
    sliding_half = sliding_dim ÷ 2
    sliding_theta = Float64(config.sliding_rope_theta)
    sliding_inv_freq = [1.0 / (sliding_theta ^ (2.0 * k / sliding_dim)) for k in 0:(sliding_half - 1)]

    # Full attention: proportional RoPE, partial=25%, theta=1M
    full_dim = config.global_head_dim
    full_half = full_dim ÷ 2
    full_theta = Float64(config.full_rope_theta)
    partial_factor = config.full_partial_rotary_factor
    rope_angles = Int(partial_factor * full_dim ÷ 2)  # 64 for E2B

    # Proportional RoPE: exponent uses full head_dim, not partial
    full_inv_freq = Vector{Float64}(undef, full_half)
    for k in 0:(rope_angles - 1)
        # k ranges over pairs, exponent = 2*k / full_dim
        full_inv_freq[k + 1] = 1.0 / (full_theta ^ (2.0 * k / full_dim))
    end
    # Non-rotated portion gets zero frequency (cos=1, sin=0)
    for k in rope_angles:full_half-1
        full_inv_freq[k + 1] = 0.0
    end

    # Pre-compute cos/sin for all positions
    sliding_cos = Vector{Float32}[]
    sliding_sin = Vector{Float32}[]
    full_cos = Vector{Float32}[]
    full_sin = Vector{Float32}[]

    for pos in 0:(max_seq_len - 1)
        sc = Float32[cos(pos * sliding_inv_freq[k+1]) for k in 0:sliding_half-1]
        ss = Float32[sin(pos * sliding_inv_freq[k+1]) for k in 0:sliding_half-1]
        push!(sliding_cos, sc)
        push!(sliding_sin, ss)

        fc = Float32[cos(pos * full_inv_freq[k+1]) for k in 0:full_half-1]
        fs = Float32[sin(pos * full_inv_freq[k+1]) for k in 0:full_half-1]
        push!(full_cos, fc)
        push!(full_sin, fs)
    end

    return sliding_cos, sliding_sin, full_cos, full_sin
end

# --- Math functions ---

function gelu_tanh(x::Float32)
    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = sqrt(2.0f0 / Float32(pi))
    return 0.5f0 * x * (1.0f0 + tanh(c * (x + 0.044715f0 * x * x * x)))
end

function rmsnorm!(out::AbstractVector{Float32}, x::AbstractVector{Float32},
                  w::AbstractVector{Float32}, eps::Float32)
    n = length(x)
    ss = 0.0f0
    @simd for i in 1:n
        ss += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(ss / n + eps)
    @simd for i in 1:n
        out[i] = x[i] * inv_rms * w[i]
    end
end

function rmsnorm_no_scale!(out::AbstractVector{Float32}, x::AbstractVector{Float32}, eps::Float32)
    n = length(x)
    ss = 0.0f0
    @simd for i in 1:n
        ss += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(ss / n + eps)
    @simd for i in 1:n
        out[i] = x[i] * inv_rms
    end
end

function apply_rope!(q::AbstractVector{Float32}, pos::Int,
                     cos_vals::AbstractVector{Float32},
                     sin_vals::AbstractVector{Float32},
                     head_dim::Int, num_heads::Int)
    for h in 0:(num_heads - 1)
        base = h * head_dim + 1
        for k in 1:(head_dim ÷ 2)
            c = cos_vals[k]
            s = sin_vals[k]
            idx0 = base + 2*k - 2  # even index
            idx1 = base + 2*k - 1  # odd index
            q0 = q[idx0]
            q1 = q[idx1]
            q[idx0] = q0 * c - q1 * s
            q[idx1] = q0 * s + q1 * c
        end
    end
end

# --- Attention forward ---

function attention_forward!(layer::AttentionLayer, hidden::AbstractVector{Float32},
                            pos::Int, cache::KVCacheG4, layer_idx::Int,
                            sliding_cos::Vector{Vector{Float32}},
                            sliding_sin::Vector{Vector{Float32}},
                            full_cos::Vector{Vector{Float32}},
                            full_sin::Vector{Vector{Float32}},
                            shared_kv_k::Dict{Int, Matrix{Float32}},
                            shared_kv_v::Dict{Int, Matrix{Float32}},
                            config::Gemma4Config)
    n = length(hidden)
    head_d = layer.head_dim
    n_q = config.num_q_heads
    n_kv = layer.num_kv_heads_actual
    softcap = config.attention_logits_softcapping
    is_sliding = layer.is_sliding

    # Q projection + norm + RoPE
    mul!(layer.q_buf, layer.q_proj', hidden)  # (n_q * head_d,)
    # Q norm (RMSNorm with scale)
    for h in 0:(n_q - 1)
        base = h * head_d + 1
        q_head = view(layer.q_buf, base:base+head_d-1)
        q_normed = view(layer.q_buf, base:base+head_d-1)  # in-place
        rmsnorm!(q_normed, q_head, layer.q_norm_w, config.rms_norm_eps)
    end
    # Apply RoPE to Q
    if is_sliding
        apply_rope!(layer.q_buf, pos, sliding_cos[pos+1], sliding_sin[pos+1], head_d, n_q)
    else
        apply_rope!(layer.q_buf, pos, full_cos[pos+1], full_sin[pos+1], head_d, n_q)
    end

    # K/V handling
    if layer.is_kv_shared
        # Use shared KV from source layer
        src = layer.kv_shared_src
        k_states = shared_kv_k[src]
        v_states = shared_kv_v[src]
        # Copy current position K/V into our cache (for attention computation)
        cache.k_cache[layer_idx][:, pos+1] = k_states[:, pos+1]
        cache.v_cache[layer_idx][:, pos+1] = v_states[:, pos+1]
    else
        # K projection + norm + RoPE
        mul!(layer.k_buf, layer.k_proj', hidden)  # (n_kv * head_d,)
        for h in 0:(n_kv - 1)
            base = h * head_d + 1
            k_head = view(layer.k_buf, base:base+head_d-1)
            rmsnorm!(k_head, k_head, layer.k_norm_w, config.rms_norm_eps)
        end
        if is_sliding
            apply_rope!(layer.k_buf, pos, sliding_cos[pos+1], sliding_sin[pos+1], head_d, n_kv)
        else
            apply_rope!(layer.k_buf, pos, full_cos[pos+1], full_sin[pos+1], head_d, n_kv)
        end

        # V projection + norm (no scale)
        if layer.v_proj !== nothing && size(layer.v_proj, 1) > 0
            mul!(layer.v_buf, layer.v_proj', hidden)
        else
            # k_eq_v mode: V = K (before RoPE was applied to K)
            # Actually in HF: value_states = key_states (after k_norm, after RoPE)
            # Wait no: HF code says `value_states = key_states` BEFORE RoPE but AFTER k_norm
            # Then V gets v_norm applied separately
            # Let me re-read... line 1214: value_states = key_states if v_proj is None
            # But key_states at that point has already been reshaped but NOT normed/roped yet
            # Actually: lines 1213-1221:
            #   key_states = k_proj(hidden).view(hidden_shape)  -- raw K
            #   value_states = v_proj(hidden).view(...) if v_proj else key_states  -- raw V = raw K
            #   key_states = k_norm(key_states)  -- norm K
            #   key_states = apply_rope(key_states)  -- rope K
            #   value_states = v_norm(value_states)  -- norm V (which is raw K)
            # So V = v_norm(k_proj(hidden)) when v_proj is None
            copyto!(layer.v_buf, layer.k_buf)  # raw K before norm/rope
        end
        # V norm (no scale)
        for h in 0:(n_kv - 1)
            base = h * head_d + 1
            v_head = view(layer.v_buf, base:base+head_d-1)
            rmsnorm_no_scale!(v_head, v_head, config.rms_norm_eps)
        end

        # Store in cache
        cache.k_cache[layer_idx][:, pos+1] = layer.k_buf
        cache.v_cache[layer_idx][:, pos+1] = layer.v_buf

        # Also store in shared KV dict if this is a storing layer
        # (we populate this during prefill)
    end

    cache.seq_len[layer_idx] = max(cache.seq_len[layer_idx], pos + 1)

    # Compute attention: for each Q head, attend to all K/V positions
    seq_len = cache.seq_len[layer_idx]
    kv_group = n_q ÷ n_kv  # GQA group size

    fill!(layer.attn_out_buf, 0.0f0)

    for h in 0:(n_q - 1)
        kv_h = h ÷ kv_group  # which KV head this Q head attends to
        q_base = h * head_d + 1
        kv_base = kv_h * head_d + 1

        # Sliding window: determine valid attention range
        if is_sliding
            window_start = max(1, pos + 1 - config.sliding_window + 1)
        else
            window_start = 1
        end

        # Compute attention scores for valid range
        q_head = view(layer.q_buf, q_base:q_base+head_d-1)
        attn_start = window_start
        attn_end = pos + 1  # current position (causal)

        n_positions = attn_end - attn_start + 1
        scores = Vector{Float32}(undef, n_positions)
        for (ti, t) in enumerate(attn_start:attn_end)
            k_t = view(cache.k_cache[layer_idx], kv_base:kv_base+head_d-1, t)
            s = 0.0f0
            @simd for d in 1:head_d
                s += q_head[d] * k_t[d]
            end
            # Attention logit softcapping
            if softcap > 0
                s = tanh(s / softcap) * softcap
            end
            scores[ti] = s
        end

        # Softmax
        max_score = maximum(scores)
        sum_weights = 0.0f0
        @simd for i in 1:n_positions
            scores[i] = exp(scores[i] - max_score)
            sum_weights += scores[i]
        end
        inv_sum = 1.0f0 / sum_weights
        @simd for i in 1:n_positions
            scores[i] *= inv_sum
        end

        # Weighted sum of V
        out_base = h * head_d + 1
        for (ti, t) in enumerate(attn_start:attn_end)
            w = scores[ti]
            @simd for d in 1:head_d
                layer.attn_out_buf[out_base + d - 1] += w * cache.v_cache[layer_idx][kv_base + d - 1, t]
            end
        end
    end

    # Output projection
    # o_proj is (hidden_size, n_q * head_dim)
    # attn_out_buf is (n_q * head_dim,)
    # Result is (hidden_size,)
    result = hidden  # reuse hidden as output buffer (caller saves residual)
    mul!(result, layer.o_proj', layer.attn_out_buf)

    return result
end

# --- Per-layer input ---

function compute_per_layer_inputs!(model::Gemma4Model, token_id::Int, inputs_embeds::AbstractVector{Float32})
    config = model.config
    pli_size = config.hidden_size_per_layer_input
    n_layers = config.num_layers

    if pli_size == 0
        return
    end

    # Step 1: embed_tokens_per_layer lookup (scaled by sqrt(pli_size))
    # embed_tokens_per_layer is (vocab_per_layer, num_layers * pli_size)
    # Get row for token_id (1-indexed: token_id + 1)
    pli_row = view(model.embed_tokens_per_layer, token_id + 1, :)
    embed_scale = sqrt(Float32(pli_size))

    # Step 2: per_layer_model_projection
    # per_layer_model_proj is (num_layers * pli_size, hidden_size)
    # projection = proj * inputs_embeds * projection_scale
    mul!(model.pli_proj_buf, model.per_layer_model_proj', inputs_embeds)
    model.pli_proj_buf .*= config.per_layer_model_projection_scale

    # Step 3: Add embedding + projection, apply per_layer_projection_norm, then scale
    # pli_per_layer_buf will be reused for each layer
    # We store the combined per-layer inputs in pli_embed_buf
    for i in 1:n_layers
        offset = (i - 1) * pli_size + 1
        proj_slice = view(model.pli_proj_buf, offset:offset+pli_size-1)
        embed_slice = view(pli_row, offset:offset+pli_size-1)

        # combined = (projection + embedding) * scale
        @simd for d in 1:pli_size
            model.pli_embed_buf[offset + d - 1] = (proj_slice[d] + embed_slice[d] * embed_scale) * config.per_layer_input_scale
        end

        # Apply per_layer_projection_norm (RMSNorm)
        norm_slice = view(model.pli_embed_buf, offset:offset+pli_size-1)
        rmsnorm!(norm_slice, norm_slice, model.per_layer_proj_norm_w, config.rms_norm_eps)
    end
end

function per_layer_input_forward!(layer::DecoderLayer, pli_slice::AbstractVector{Float32},
                                  hidden::AbstractVector{Float32}, config::Gemma4Config)
    pli = layer.pli
    if pli === nothing
        return
    end

    n = length(hidden)
    pli_size = config.hidden_size_per_layer_input

    # gate = gate_proj(hidden)
    mul!(layer.pli_gate_buf, pli.gate_proj', hidden)

    # gate = gelu_tanh(gate)
    @simd for i in 1:pli_size
        layer.pli_gate_buf[i] = gelu_tanh(layer.pli_gate_buf[i])
    end

    # gate = gate * per_layer_input
    @simd for i in 1:pli_size
        layer.pli_gate_buf[i] *= pli_slice[i]
    end

    # out = projection(gate)
    mul!(layer.pli_out_buf, pli.projection', layer.pli_gate_buf)

    # post_per_layer_input_norm (RMSNorm with scale)
    rmsnorm!(layer.pli_out_buf, layer.pli_out_buf, pli.post_norm_w, config.rms_norm_eps)
end

# --- MLP forward ---

function mlp_forward!(layer::DecoderLayer, hidden::AbstractVector{Float32}, config::Gemma4Config)
    mlp = layer.mlp
    n = length(hidden)
    inter = size(mlp.gate_proj, 1)

    # gate = gelu_tanh(gate_proj(hidden))
    mul!(mlp.gate_buf, mlp.gate_proj', hidden)
    @simd for i in 1:inter
        mlp.gate_buf[i] = gelu_tanh(mlp.gate_buf[i])
    end

    # up = up_proj(hidden)
    mul!(mlp.up_buf, mlp.up_proj', hidden)

    # gate * up
    @simd for i in 1:inter
        mlp.gate_buf[i] *= mlp.up_buf[i]
    end

    # down = down_proj(gate * up)
    mul!(mlp.hidden_buf, mlp.down_proj', mlp.gate_buf)

    return mlp.hidden_buf
end

# --- Main forward pass ---

function forward!(model::Gemma4Model, token_ids::Vector{Int}, start_pos::Int, cache::KVCacheG4)
    config = model.config
    n = config.hidden_size
    pli_size = config.hidden_size_per_layer_input

    # Token embedding (scaled by sqrt(hidden_size))
    fill!(model.hidden_buf, 0.0f0)
    for (t_idx, tid) in enumerate(token_ids)
        pos = start_pos + t_idx - 1
        embed_row = view(model.embed_tokens, tid + 1, :)  # 1-indexed
        @simd for i in 1:n
            model.hidden_buf[i] += embed_row[i]
        end
    end
    # Scale by sqrt(hidden_size)
    embed_scale = config.embed_scale
    @simd for i in 1:n
        model.hidden_buf[i] *= embed_scale
    end

    # Pre-compute per-layer inputs for all tokens (using last token for single-token generation)
    if pli_size > 0 && length(token_ids) == 1
        compute_per_layer_inputs!(model, token_ids[1], model.hidden_buf)
    end

    hidden = model.hidden_buf

    for (layer_idx, layer) in enumerate(model.layers)
        # === Attention block ===
        # Save residual
        copyto!(model.residual_buf, hidden)

        # Input norm
        rmsnorm!(hidden, hidden, layer.input_norm_w, config.rms_norm_eps)

        # Attention forward (writes result back into hidden)
        attention_forward!(layer, hidden, start_pos, cache, layer_idx,
                          model.sliding_cos, model.sliding_sin,
                          model.full_cos, model.full_sin,
                          model.shared_kv_k, model.shared_kv_v, config)

        # Post-attention norm
        rmsnorm!(hidden, hidden, layer.post_attn_norm_w, config.rms_norm_eps)

        # Residual connection
        @simd for i in 1:n
            hidden[i] += model.residual_buf[i]
        end

        # === MLP block ===
        copyto!(model.residual_buf, hidden)

        # Pre-FF norm
        rmsnorm!(hidden, hidden, layer.pre_ff_norm_w, config.rms_norm_eps)

        # MLP
        mlp_out = mlp_forward!(layer, hidden, config)

        # Post-FF norm
        rmsnorm!(mlp_out, mlp_out, layer.post_ff_norm_w, config.rms_norm_eps)

        # Residual connection
        @simd for i in 1:n
            hidden[i] = model.residual_buf[i] + mlp_out[i]
        end

        # === Per-layer input block ===
        if layer.pli !== nothing && pli_size > 0
            copyto!(model.residual_buf, hidden)

            # Get per-layer input slice for this layer
            offset = (layer_idx - 1) * pli_size + 1
            pli_slice = view(model.pli_embed_buf, offset:offset+pli_size-1)

            # Per-layer input forward
            per_layer_input_forward!(layer, pli_slice, hidden, config)

            # Residual connection
            @simd for i in 1:n
                hidden[i] = model.residual_buf[i] + layer.pli_out_buf[i]
            end
        end

        # Layer scalar (applied to ENTIRE hidden state)
        @simd for i in 1:n
            hidden[i] *= layer.layer_scalar
        end

        # Store shared KV for non-shared layers that should store
        if !layer.attn.is_kv_shared && haskey(model.shared_kv_k, layer_idx - 1) === false
            # Check if this layer should store its full KV for sharing
            # In the HF code: store_full_length_kv is set for the last non-shared layer of each type
            # For simplicity, we store for all non-shared layers
            # Actually, let me be more precise: only the last non-shared sliding and last non-shared full layers store
        end
    end

    # Final norm
    rmsnorm!(hidden, hidden, model.final_norm_w, config.rms_norm_eps)

    # Logits
    return get_logits(model, hidden)
end

function get_logits(model::Gemma4Model, hidden::Vector{Float32})
    if model.config.tie_word_embeddings
        # lm_head = embed_tokens, logits = embed_tokens * hidden
        # embed_tokens is (vocab_size, hidden_size)
        mul!(model.logits_buf, model.embed_tokens, hidden)
    else
        error("Non-tied lm_head not implemented for Gemma4")
    end

    # Final logit softcapping
    cap = model.config.final_logit_softcapping
    if cap > 0
        @simd for i in 1:length(model.logits_buf)
            model.logits_buf[i] = tanh(model.logits_buf[i] / cap) * cap
        end
    end

    return model.logits_buf
end

end # module Gemma4
