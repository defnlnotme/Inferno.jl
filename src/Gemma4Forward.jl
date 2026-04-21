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

function precompute_freqs(head_dim::Int, max_seq_len::Int, theta::Float64; partial_rotary_factor::Float64=1.0)
    rotary_dim = Int(head_dim * partial_rotary_factor)
    rotary_dim = max(rotary_dim, 2)
    half_dim = rotary_dim ÷ 2
    freqs = Float64[1.0 / (theta ^ (2k / rotary_dim)) for k in 0:(half_dim - 1)]
    t = Float64[Float64(i) for i in 0:(max_seq_len - 1)]
    freqs_t = t .* freqs'  # [seq_len, half_dim]
    cos_freqs = Float32.(cos.(freqs_t))
    sin_freqs = Float32.(sin.(freqs_t))
    return cos_freqs, sin_freqs
end

function apply_rope!(x::Vector{Float32}, pos::Int, cos_table::Matrix{Float32}, sin_table::Matrix{Float32}, head_dim::Int, n_heads::Int)
    half_dim = size(cos_table, 2)
    for h in 0:(n_heads - 1)
        base = h * head_dim + 1
        for k in 0:(half_dim - 1)
            x0 = x[base + k]
            x1 = x[base + half_dim + k]
            c = cos_table[pos + 1, k + 1]
            s = sin_table[pos + 1, k + 1]
            x[base + k]          = x0 * c - x1 * s
            x[base + half_dim + k] = x0 * s + x1 * c
        end
    end
end

# ============================================================
# RMSNorm (Gemma4: no +1 bias, standard RMSNorm)
# ============================================================

function rmsnorm_g4!(out::Vector{Float32}, x::Vector{Float32}, w::Vector{Float32}, eps::Float32)
    n = length(x)
    ss = eps
    @simd for i in 1:n
        ss += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(ss / n)
    @simd for i in 1:n
        out[i] = w[i] * x[i] * inv_rms
    end
end

function rmsnorm_g4_inplace!(x::Vector{Float32}, w::Vector{Float32}, eps::Float32)
    n = length(x)
    ss = eps
    @simd for i in 1:n
        ss += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(ss / n)
    @simd for i in 1:n
        x[i] = w[i] * x[i] * inv_rms
    end
end

function rmsnorm_g4_noscale!(out::Vector{Float32}, x::Vector{Float32}, eps::Float32)
    n = length(x)
    ss = eps
    @simd for i in 1:n
        ss += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(ss / n)
    @simd for i in 1:n
        out[i] = x[i] * inv_rms
    end
end

# ============================================================
# GELU Tanh Activation (approximation)
# ============================================================

function gelu_tanh(x::Float32)
    c = 0.7978845608028654f0  # sqrt(2/pi)
    inner = c * (x + 0.044715f0 * x * x)
    return 0.5f0 * x * (1.0f0 + tanh(inner))
end

# ============================================================
# Attention
# ============================================================

function attention_forward!(
    q_buf::Vector{Float32}, k_buf::Vector{Float32}, v_buf::Vector{Float32},
    attn_out::Vector{Float32},
    hidden::Vector{Float32},
    q_proj::Matrix{Float32}, k_proj::Matrix{Float32}, v_proj::Matrix{Float32},
    o_proj::Matrix{Float32},
    q_norm_w::Vector{Float32}, k_norm_w::Vector{Float32},
    num_q_heads::Int, num_kv_heads::Int, head_dim::Int,
    cos_table::Matrix{Float32}, sin_table::Matrix{Float32},
    pos::Int, cache::KVCacheG4,
    is_sliding::Bool, sliding_window::Int,
    q_normed::Vector{Float32}, k_normed::Vector{Float32}, v_normed::Vector{Float32},
    norm_buf::Vector{Float32},
    eps::Float32
)
    hidden_dim = length(hidden)
    kv_dim = num_kv_heads * head_dim
    q_dim = num_q_heads * head_dim
    
    # Q/K/V projections
    mul!(q_buf, q_proj', hidden)  # q_proj is [hidden, q_dim], q_proj' * hidden
    mul!(k_buf, k_proj', hidden)
    mul!(v_buf, v_proj', hidden)
    
    # Q norm (RMSNorm with scale)
    for h in 0:(num_q_heads - 1)
        off = h * head_dim
        rmsnorm_g4!(view(q_normed, off+1:off+head_dim), view(q_buf, off+1:off+head_dim), q_norm_w, eps)
    end
    
    # K norm (RMSNorm with scale)
    for h in 0:(num_kv_heads - 1)
        off = h * head_dim
        rmsnorm_g4!(view(k_normed, off+1:off+head_dim), view(k_buf, off+1:off+head_dim), k_norm_w, eps)
    end
    
    # V norm (RMSNorm without scale)
    for h in 0:(num_kv_heads - 1)
        off = h * head_dim
        rmsnorm_g4_noscale!(view(v_normed, off+1:off+head_dim), view(v_buf, off+1:off+head_dim), eps)
    end
    
    # Apply RoPE to Q and K
    apply_rope!(q_normed, pos, cos_table, sin_table, head_dim, num_q_heads)
    apply_rope!(k_normed, pos, cos_table, sin_table, head_dim, num_kv_heads)
    
    # Store K,V in cache
    cache_len = cache.len[1]
    for i in 1:kv_dim
        cache.k[cache_len + 1, i] = k_normed[i]
        cache.v[cache_len + 1, i] = v_normed[i]
    end
    cache.len[1] = cache_len + 1
    cur_len = cache.len[1]
    
    # Attention: Q * K^T / sqrt(head_dim)
    # For sliding window, only attend to last sliding_window positions
    start_pos = is_sliding ? max(1, cur_len - sliding_window) : 1
    num_groups = num_q_heads ÷ num_kv_heads
    
    fill!(attn_out, 0.0f0)
    
    for h in 0:(num_q_heads - 1)
        kv_h = h ÷ num_groups  # which KV head this Q head attends to
        q_off = h * head_dim
        
        # Compute attention scores for this head
        max_score = -1.0f10
        score_sum = 0.0f0
        
        for pos_idx in start_pos:cur_len
            score = 0.0f0
            k_off = (kv_h) * head_dim
            @simd for d in 1:head_dim
                score += q_normed[q_off + d] * cache.k[pos_idx, k_off + d]
            end
            score /= sqrt(Float32(head_dim))
            
            # Softmax (online)
            if score > max_score
                # Rescale
                old_max = max_score
                max_score = score
                if score_sum > 0.0f0
                    correction = exp(old_max - max_score)
                    score_sum *= correction
                    for d in 1:head_dim
                        attn_out[q_off + d] *= correction
                    end
                end
            end
            weight = exp(score - max_score)
            score_sum += weight
            
            v_off = (kv_h) * head_dim
            @simd for d in 1:head_dim
                attn_out[q_off + d] += weight * cache.v[pos_idx, v_off + d]
            end
        end
        
        # Normalize
        if score_sum > 0.0f0
            inv_sum = 1.0f0 / score_sum
            @simd for d in 1:head_dim
                attn_out[q_off + d] *= inv_sum
            end
        end
    end
    
    # Output projection: o_proj * attn_out
    mul!(norm_buf, o_proj', attn_out)
    copyto!(attn_out, norm_buf)
end

# Attention with shared KV
function attention_forward_shared_kv!(
    attn_out::Vector{Float32},
    hidden::Vector{Float32},
    q_proj::Matrix{Float32}, o_proj::Matrix{Float32},
    q_norm_w::Vector{Float32},
    num_q_heads::Int, num_kv_heads::Int, head_dim::Int,
    cos_table::Matrix{Float32}, sin_table::Matrix{Float32},
    pos::Int,
    shared_k::Matrix{Float32}, shared_v::Matrix{Float32}, shared_len::Int,
    is_sliding::Bool, sliding_window::Int,
    q_buf::Vector{Float32}, q_normed::Vector{Float32},
    norm_buf::Vector{Float32}
)
    hidden_dim = length(hidden)
    q_dim = num_q_heads * head_dim
    
    # Q projection only
    mul!(q_buf, q_proj', hidden)
    
    # Q norm
    for h in 0:(num_q_heads - 1)
        off = h * head_dim
        rmsnorm_g4!(view(q_normed, off+1:off+head_dim), view(q_buf, off+1:off+head_dim), q_norm_w, 1.0f-6)
    end
    
    # Apply RoPE to Q
    apply_rope!(q_normed, pos, cos_table, sin_table, head_dim, num_q_heads)
    
    # Attention using shared KV
    start_pos = is_sliding ? max(1, shared_len - sliding_window) : 1
    num_groups = num_q_heads ÷ num_kv_heads
    
    fill!(attn_out, 0.0f0)
    
    for h in 0:(num_q_heads - 1)
        kv_h = h ÷ num_groups
        q_off = h * head_dim
        
        max_score = -1.0f10
        score_sum = 0.0f0
        
        for pos_idx in start_pos:shared_len
            score = 0.0f0
            k_off = kv_h * head_dim
            @simd for d in 1:head_dim
                score += q_normed[q_off + d] * shared_k[pos_idx, k_off + d]
            end
            score /= sqrt(Float32(head_dim))
            
            if score > max_score
                old_max = max_score
                max_score = score
                if score_sum > 0.0f0
                    correction = exp(old_max - max_score)
                    score_sum *= correction
                    for d in 1:head_dim
                        attn_out[q_off + d] *= correction
                    end
                end
            end
            weight = exp(score - max_score)
            score_sum += weight
            
            v_off = kv_h * head_dim
            @simd for d in 1:head_dim
                attn_out[q_off + d] += weight * shared_v[pos_idx, v_off + d]
            end
        end
        
        if score_sum > 0.0f0
            inv_sum = 1.0f0 / score_sum
            @simd for d in 1:head_dim
                attn_out[q_off + d] *= inv_sum
            end
        end
    end
    
    mul!(norm_buf, o_proj', attn_out)
    copyto!(attn_out, norm_buf)
end

# ============================================================
# MLP Forward
# ============================================================

function mlp_forward!(
    out::Vector{Float32},
    x::Vector{Float32},
    gate_proj::Matrix{Float32}, up_proj::Matrix{Float32}, down_proj::Matrix{Float32},
    gate_buf::Vector{Float32}, up_buf::Vector{Float32}
)
    # gate = GELU(gate_proj * x)
    mul!(gate_buf, gate_proj', x)
    @simd for i in eachindex(gate_buf)
        gate_buf[i] = gelu_tanh(gate_buf[i])
    end
    
    # up = up_proj * x
    mul!(up_buf, up_proj', x)
    
    # intermediate = gate * up
    @simd for i in eachindex(gate_buf)
        gate_buf[i] = gate_buf[i] * up_buf[i]
    end
    
    # down = down_proj * intermediate
    mul!(out, down_proj', gate_buf)
end

# ============================================================
# Decoder Layer Forward
# ============================================================

function layer_forward!(
    layer::Gemma4DecoderLayerCPU,
    hidden::Vector{Float32},
    per_layer_embed::Union{Vector{Float32}, Nothing},
    cos_table::Matrix{Float32}, sin_table::Matrix{Float32},
    pos::Int,
    cache::KVCacheG4,
    shared_kv_k::Dict{Int, Matrix{Float32}},
    shared_kv_v::Dict{Int, Matrix{Float32}},
    eps::Float32
)
    H = layer.hidden_size
    
    # === Attention block ===
    # residual = hidden
    copyto!(layer.residual, hidden)
    
    # input_layernorm
    rmsnorm_g4_inplace!(hidden, layer.input_norm_w, eps)
    
    attn = layer.attention
    if attn.is_kv_shared
        # Shared KV: use stored KV from source layer
        src_idx = attn.kv_shared_layer_idx
        src_k = shared_kv_k[src_idx]
        src_v = shared_kv_v[src_idx]
        src_len = size(src_k, 1)  # This should track the actual length
        
        attention_forward_shared_kv!(
            attn.attn_buf, hidden,
            attn.q_proj, attn.o_proj,
            attn.q_norm_w,
            attn.num_q_heads, attn.num_kv_heads, attn.head_dim,
            cos_table, sin_table, pos,
            src_k, src_v, src_len,
            attn.is_sliding, attn.sliding_window,
            attn.q_buf, attn.q_normed,
            attn.v_normed  # reuse as temp buffer
        )
    else
        attention_forward!(
            attn.q_buf, attn.k_buf, attn.v_buf,
            attn.attn_buf,
            hidden,
            attn.q_proj, attn.k_proj, attn.v_proj,
            attn.o_proj,
            attn.q_norm_w, attn.k_norm_w,
            attn.num_q_heads, attn.num_kv_heads, attn.head_dim,
            cos_table, sin_table, pos, cache,
            attn.is_sliding, attn.sliding_window,
            attn.q_normed, attn.k_normed, attn.v_normed,
            layer.norm_buf1,
            eps
        )
        
        # Store KV for sharing if this is the "storing" layer
        if attn.store_full_kv
            shared_kv_k[attn.layer_idx] = copy(cache.k[1:cache.len[1], :])
            shared_kv_v[attn.layer_idx] = copy(cache.v[1:cache.len[1], :])
        end
    end
    
    # post_attention_layernorm
    rmsnorm_g4!(layer.norm_buf1, attn.attn_buf, layer.post_attn_norm_w, eps)
    
    # residual + norm(attn_out)
    @simd for i in 1:H
        hidden[i] = layer.residual[i] + layer.norm_buf1[i]
    end
    
    # === MLP block ===
    copyto!(layer.residual, hidden)
    
    # pre_feedforward_layernorm
    rmsnorm_g4_inplace!(hidden, layer.pre_ff_norm_w, eps)
    
    mlp = layer.mlp
    mlp_forward!(layer.norm_buf1, hidden, mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.gate_buf, mlp.up_buf)
    
    # post_feedforward_layernorm
    rmsnorm_g4!(layer.norm_buf2, layer.norm_buf1, layer.post_ff_norm_w, eps)
    
    # residual + norm(mlp_out)
    @simd for i in 1:H
        hidden[i] = layer.residual[i] + layer.norm_buf2[i]
    end
    
 # === Per-layer input ===
 if per_layer_embed !== nothing && layer.per_layer_input !== nothing
 pli = layer.per_layer_input
 copyto!(layer.residual, hidden)
 
 # gate = GELU(gate_proj * hidden) -- GELU tanh, NOT SiLU!
 mul!(pli.gate_buf, pli.gate_proj', hidden)
 @simd for i in eachindex(pli.gate_buf)
 pli.gate_buf[i] = gelu_tanh(pli.gate_buf[i])
 end
 
 # gate * per_layer_embed (element-wise)
 @simd for i in eachindex(pli.gate_buf)
 pli.gate_buf[i] = pli.gate_buf[i] * per_layer_embed[i]
 end
 
 # projection: per_layer_dim -> hidden_size
 mul!(pli.proj_buf, pli.projection', pli.gate_buf)
 
 # post norm (RMSNorm with scale)
 rmsnorm_g4!(layer.norm_buf1, pli.proj_buf, pli.post_norm_w, eps)
 
 # residual + per_layer
 @simd for i in 1:H
 hidden[i] = layer.residual[i] + layer.norm_buf1[i]
 end
 end
    
    # layer_scalar
    if layer.layer_scalar != 1.0f0
        @simd for i in 1:H
            hidden[i] *= layer.layer_scalar
        end
    end
    
    return hidden
end

# ============================================================
# Full Model Forward Pass
# ============================================================

function forward!(
    model::Gemma4ModelCPU,
    token_ids::Vector{Int},
    start_pos::Int
)
    H = model.hidden_size
    num_tokens = length(token_ids)
    
    # Embedding lookup with scaling
    hidden = zeros(Float32, H)
    for tid in token_ids
        # tid is 0-indexed from tokenizer, but our matrix is 1-indexed
        embed_idx = tid + 1
        if embed_idx >= 1 && embed_idx <= size(model.embed_tokens, 1)
            @simd for d in 1:H
                hidden[d] += model.embed_tokens[embed_idx, d]
            end
        end
    end
    # Scale by sqrt(hidden_size) per Gemma4 convention
    @simd for d in 1:H
        hidden[d] *= model.embed_scale
    end
    
 # Compute per-layer inputs if applicable
 # HF: per_layer_inputs = (projection_norm(model_projection(embed) * scale) + embed_per_layer) * input_scale
 per_layer_inputs = Vector{Union{Vector{Float32}, Nothing}}(nothing, model.num_layers)
 if model.per_layer_dim > 0 && model.embed_per_layer !== nothing
 # Step 1: Get per-layer raw embeddings for this token
 # embed_per_layer shape: [vocab_per_layer, num_layers * per_layer_dim]
 # Already scaled by embed_tokens_per_layer scale (sqrt(per_layer_dim))
 for tid in token_ids
 embed_idx = tid + 1
 raw = view(model.embed_per_layer, embed_idx, :) # [num_layers * per_layer_dim]
 for layer_i in 1:model.num_layers
 off = (layer_i - 1) * model.per_layer_dim
 if per_layer_inputs[layer_i] === nothing
 per_layer_inputs[layer_i] = zeros(Float32, model.per_layer_dim)
 end
 for d in 1:model.per_layer_dim
 per_layer_inputs[layer_i][d] += raw[off + d]
 end
 end
 end
 
 # Step 2: Apply per_layer_model_projection to the main embedding
 # This projects from hidden_size -> num_layers * per_layer_dim
 pli = first(model.layers).per_layer_input # get model_projection from any layer (shared)
 pli_total_dim = model.num_layers * model.per_layer_dim
 mul!(reshape(pli.model_proj_buf, pli_total_dim, 1), pli.model_projection', reshape(hidden, H, 1))
 
 # Scale by 1/sqrt(hidden_size)
 mps = pli.model_projection_scale
 @simd for i in 1:pli_total_dim
 pli.model_proj_buf[i] *= mps
 end
 
 # Step 3: Reshape model_projection output and apply per_layer_projection_norm
 # norm operates on each per_layer_dim slice independently
 for layer_i in 1:model.num_layers
 off = (layer_i - 1) * model.per_layer_dim
 proj_slice = view(pli.model_proj_buf, off+1:off+model.per_layer_dim)
 
 # Apply RMSNorm to model_projection slice (per_layer_projection_norm has shape [pli_total_dim])
 norm_w_slice = view(pli.post_norm_w, off+1:off+model.per_layer_dim)
 
 # RMSNorm on the projection slice
 ss = model.rms_norm_eps
 @simd for d in 1:model.per_layer_dim
 ss += proj_slice[d] * proj_slice[d]
 end
 inv_rms = 1.0f0 / sqrt(ss / model.per_layer_dim)
 @simd for d in 1:model.per_layer_dim
 proj_slice[d] = proj_slice[d] * inv_rms * norm_w_slice[d]
 end
 
 # Step 4: Combine projection with per-layer embeddings: (projection + embed) * input_scale
 pis = pli.per_layer_input_scale
 @simd for d in 1:model.per_layer_dim
 per_layer_inputs[layer_i][d] = (proj_slice[d] + per_layer_inputs[layer_i][d]) * pis
 end
 end
 end
    
    # Process through each layer
    for (li, layer) in enumerate(model.layers)
        pos = start_pos
        lt = model.layer_types[li]
        
        if lt == :sliding
            cos_t = model.sliding_cos
            sin_t = model.sliding_sin
        else
            cos_t = model.full_cos
            sin_t = model.full_sin
        end
        
        layer_forward!(
            layer, hidden, per_layer_inputs[li],
            cos_t, sin_t, pos,
            model.kv_caches[li],
            model.shared_kv_k, model.shared_kv_v,
            model.rms_norm_eps
        )
    end
    
    # Final RMSNorm
    rmsnorm_g4_inplace!(hidden, model.final_norm_w, model.rms_norm_eps)
    
    # LM head (logits)
    logits = model.lm_head' * hidden  # [vocab_size]
    
    # Logit softcapping
    if model.logit_softcapping > 0.0f0
        cap = model.logit_softcapping
        @simd for i in eachindex(logits)
            logits[i] = cap * tanh(logits[i] / cap)
        end
    end
    
    return logits
end

function generate_token(model::Gemma4ModelCPU, token_id::Int, pos::Int)
    forward!(model, [token_id], pos)
end

# ============================================================
# Generation Loop
# ============================================================

function generate_gemma4!(
    model::Gemma4ModelCPU,
    prompt_ids::Vector{Int};
    max_tokens::Int = 128,
    temperature::Float32 = 0.7f0,
    top_k::Int = 50,
    top_p::Float32 = 0.95f0,
    stop_tokens::Set{Int} = Set{Int}(),
    stream_callback::Union{Function, Nothing} = nothing
)
    # Reset KV caches
    for cache in model.kv_caches
        cache.len[1] = 0
    end
    # Reset shared KV
    for k in keys(model.shared_kv_k)
        delete!(model.shared_kv_k, k)
    end
    for k in keys(model.shared_kv_v)
        delete!(model.shared_kv_v, k)
    end
    
    # Prefill: process all prompt tokens
    for (i, tid) in enumerate(prompt_ids)
        logits = forward!(model, [tid], i - 1)
    end
    
    generated = Int[]
    pos = length(prompt_ids)
    
    # Sample first token from prefill logits
    next_token = sample_token(logits, temperature, top_k, top_p)
    push!(generated, next_token)
    
    if stream_callback !== nothing
        stream_callback(next_token)
    end
    
    # Autoregressive generation
    for _ in 2:max_tokens
        if next_token in stop_tokens
            break
        end
        
        logits = forward!(model, [next_token], pos)
        pos += 1
        
        next_token = sample_token(logits, temperature, top_k, top_p)
        push!(generated, next_token)
        
        if stream_callback !== nothing
            stream_callback(next_token)
        end
    end
    
    return generated
end

function sample_token(logits::Vector{Float32}, temperature::Float32, top_k::Int, top_p::Float32)
    if temperature < 0.01f0
        return argmax(logits) - 1  # 0-indexed
    end
    
    # Temperature scaling
    scaled = logits ./ temperature
    
    # Top-K filtering
    sorted_idx = sortperm(scaled, rev=true)
    if top_k > 0 && top_k < length(scaled)
        cutoff = scaled[sorted_idx[top_k + 1]]
        @simd for i in eachindex(scaled)
            if scaled[i] < cutoff
                scaled[i] = -1.0f10
            end
        end
    end
    
    # Softmax
    max_val = maximum(scaled)
    exps = exp.(scaled .- max_val)
    probs = exps ./ sum(exps)
    
    # Top-P (nucleus) filtering
    if top_p < 1.0f0
        sorted_probs = sort(probs, rev=true)
        cumsum = 0.0f0
        cutoff_idx = length(sorted_probs)
        for (i, p) in enumerate(sorted_probs)
            cumsum += p
            if cumsum > top_p
                cutoff_idx = i
                break
            end
        end
        # Zero out tokens beyond top-p
        threshold = sorted_probs[cutoff_idx]
        @simd for i in eachindex(probs)
            if probs[i] < threshold * 0.5f0  # approximate
                probs[i] = 0.0f0
            end
        end
        # Renormalize
        probs ./= sum(probs)
    end
    
    # Sample
    r = rand(Float32)
    cumsum = 0.0f0
    for i in eachindex(probs)
        cumsum += probs[i]
        if cumsum >= r
            return i - 1  # 0-indexed
        end
    end
    return argmax(probs) - 1
end

end # module
