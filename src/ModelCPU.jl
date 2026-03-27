"""
CPU-only inference backend for Inferno.jl
This module provides pure CPU implementations without GPU dependencies.
"""
module ModelCPU

using LinearAlgebra
using Statistics

export QwenConfigCPU, QwenModelCPU, KVCacheCPU, forward_cpu!, RMSNormCPU, MLPCPU, GatedDeltaNetCPU, FullAttentionCPU, DecoderLayerCPU, RotaryEmbeddingCPU
export init_kv_cache_cpu, reset_states_cpu!, softmax_sample, generate_cpu, generate_stream_cpu, stream_to_stdout_cpu

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
    partial_rotary_factor::Float32 = 0.25f0  # Only 25% of head_dim gets rotary
end

# Helper functions
sigmoid(x) = 1.0f0 / (1.0f0 + exp(-x))

# --- RMS Norm ---
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
    # Qwen3.5 uses (1 + weight) instead of just weight
    return x .* scale .* (1.0f0 .+ norm.weight)
end

function rmsnorm_cpu!(out::AbstractArray{Float32}, x::AbstractArray{Float32}, norm::RMSNormCPU)
    ss = mapreduce(abs2, +, x)
    m = ss / length(x)
    scale = 1.0f0 / sqrt(m + norm.eps)
    # Qwen3.5 uses (1 + weight) instead of just weight
    out .= x .* scale .* (1.0f0 .+ norm.weight)
    return out
end

# --- Rotary Position Embedding ---
struct RotaryEmbeddingCPU
    inv_freq::Vector{Float32}
    max_seq_len::Int
    rotary_dim::Int  # Number of dimensions that get rotary (partial rotary)
end

function RotaryEmbeddingCPU(head_dim::Int, theta::Float32 = 10000.0f0, max_seq_len::Int = 4096; rotary_dim::Int = head_dim)
    # Only compute inv_freq for the rotary dimensions
    inv_freq = Float32[1.0 / (theta ^ (2(i-1)/head_dim)) for i in 1:div(rotary_dim, 2)]
    return RotaryEmbeddingCPU(inv_freq, max_seq_len, rotary_dim)
end

function apply_rotary_emb!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU)
    head_dim, num_heads = size(x, 1), size(x, 2)
    half = div(rope.rotary_dim, 2)
    
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
    wq::Matrix{Float32}  # Projects to n_heads * head_dim * 2 (query + gate)
    wk::Matrix{Float32}  # Projects to n_kv * head_dim
    wv::Matrix{Float32}  # Projects to n_kv * head_dim
    wo::Matrix{Float32}  # Projects from n_heads * head_dim to hidden
    q_norm::RMSNormCPU
    k_norm::RMSNormCPU
    n_heads::Int
    n_kv::Int
    head_dim::Int
    scale::Float32
end

function (attn::FullAttentionCPU)(x::Vector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
    # Q, K, V projections
    qkv = attn.wq * x  # This produces n_heads * head_dim * 2 output
    k = attn.wk * x
    v = attn.wv * x
    
    # Split qkv into query and gate
    # qkv has shape (n_heads * head_dim * 2,) = (n_heads * head_dim,) + (n_heads * head_dim,)
    q_size = attn.n_heads * attn.head_dim
    query_states = qkv[1:q_size]
    gate = qkv[q_size+1:end]
    
    # Reshape to (head_dim, num_heads)
    query_states = reshape(query_states, attn.head_dim, attn.n_heads)
    k = reshape(k, attn.head_dim, attn.n_kv)
    v = reshape(v, attn.head_dim, attn.n_kv)
    
    # Apply Q/K normalization
    query_states = attn.q_norm(query_states)
    k = attn.k_norm(k)
    
    # Apply RoPE
    apply_rotary_emb!(query_states, pos, rope)
    apply_rotary_emb!(k, pos, rope)
    
    # Update KV cache
    update_kv_cache!(cache, k, v, pos)
    
    # Compute attention scores
    output = zeros(Float32, attn.n_heads * attn.head_dim)
    
    gqa_ratio = div(attn.n_heads, attn.n_kv)
    
    for h in 1:attn.n_heads
        kv_h = div(h - 1, gqa_ratio) + 1
        
        q_h = view(query_states, :, h)
        
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
    
    # Apply gate with sigmoid
    # gate is (n_heads * head_dim,) and output is (n_heads * head_dim,)
    # Apply sigmoid element-wise and multiply
    output .*= sigmoid.(gate)
    
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

# --- Sampling Functions ---

function softmax_sample(logits::Vector{Float32}; temperature::Float32=1.0f0, top_p::Float32=1.0f0, top_k::Int=0)
    # Apply temperature
    if temperature != 1.0f0
        logits = logits ./ temperature
    end
    
    # Apply top-k filtering
    if top_k > 0
        sorted_indices = sortperm(logits, rev=true)
        keep_indices = Set(sorted_indices[1:min(top_k, length(logits))])
        for i in 1:length(logits)
            if i ∉ keep_indices
                logits[i] = -Inf32
            end
        end
    end
    
    # Apply softmax
    max_logit = maximum(logits)
    exp_logits = exp.(logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0f0
        sorted_indices = sortperm(probs, rev=true)
        cumsum = 0.0f0
        keep_indices = Set{Int}()
        for idx in sorted_indices
            push!(keep_indices, idx)
            cumsum += probs[idx]
            if cumsum >= top_p
                break
            end
        end
        # Zero out probabilities for tokens not in top-p
        for i in 1:length(probs)
            if i ∉ keep_indices
                probs[i] = 0.0f0
            end
        end
        # Renormalize
        probs ./= sum(probs)
    end
    
    # Sample from distribution
    r = rand(Float32)
    cumsum = 0.0f0
    for i in 1:length(probs)
        cumsum += probs[i]
        if r <= cumsum
            return i
        end
    end
    return length(probs)
end

function apply_repetition_penalty!(logits::Vector{Float32}, token_counts::Dict{Int,Int}, penalty::Float32)
    if penalty != 1.0f0
        for (tok, count) in token_counts
            if logits[tok] > 0
                logits[tok] /= penalty^count
            else
                logits[tok] *= penalty^count
            end
        end
    end
end

# --- Generation Functions ---

"""
    generate_cpu(model, tokens, pos, caches; kwargs...)

Generate a single token from the model.

Returns: (next_token, updated_logits)
"""
function generate_cpu(model::QwenModelCPU, tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU};
    temperature::Float32=1.0f0, top_p::Float32=1.0f0, top_k::Int=0,
    repetition_penalty::Float32=1.0f0, token_counts::Dict{Int,Int}=Dict{Int,Int}())
    
    # Forward pass
    logits = forward_cpu!(model, tokens, pos, caches)
    
    # Get logits for last token
    logits_vec = vec(logits[:, end])
    
    # Apply repetition penalty
    apply_repetition_penalty!(logits_vec, token_counts, repetition_penalty)
    
    # Sample
    next_token = softmax_sample(logits_vec; temperature, top_p, top_k)
    
    return next_token, logits_vec
end

"""
    generate_stream_cpu(model, prompt_tokens; kwargs...)

Create a channel that yields decoded token strings as they are generated.

Usage:
```julia
stream = generate_stream_cpu(model, prompt_tokens; max_tokens=100)
for token_str in stream
    print(token_str)
end
```
"""
function generate_stream_cpu(model::QwenModelCPU, prompt_tokens::Vector{Int}, decode_fn::Function;
    max_tokens::Int=512,
    temperature::Float32=1.0f0,
    top_p::Float32=0.95f0,
    top_k::Int=0,
    repetition_penalty::Float32=1.0f0,
    stop_tokens::Set{Int}=Set{Int}())
    
    return Channel{String}(32) do chan
        try
            # Initialize caches and states
            caches = [init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
            reset_states_cpu!(model)
            
            # Track token counts for repetition penalty
            token_counts = Dict{Int,Int}()
            for t in prompt_tokens
                token_counts[t] = get(token_counts, t, 0) + 1
            end
            
            # Process prompt tokens
            curr_pos = 0
            if !isempty(prompt_tokens)
                _ = forward_cpu!(model, prompt_tokens[1:end-1], 0, caches)
                curr_pos = length(prompt_tokens) - 1
                
                # Generate first token from prompt context
                next_token, _ = generate_cpu(model, [prompt_tokens[end]], curr_pos, caches;
                    temperature, top_p, top_k, repetition_penalty, token_counts)
                
                curr_pos += 1
                token_counts[next_token] = get(token_counts, next_token, 0) + 1
                
                # Decode and yield
                token_str = decode_fn([next_token])
                put!(chan, token_str)
                
                last_token = next_token
                
                # Generate remaining tokens
                for _ in 2:max_tokens
                    if last_token in stop_tokens
                        break
                    end
                    
                    next_token, _ = generate_cpu(model, [last_token], curr_pos, caches;
                        temperature, top_p, top_k, repetition_penalty, token_counts)
                    
                    curr_pos += 1
                    token_counts[next_token] = get(token_counts, next_token, 0) + 1
                    
                    token_str = decode_fn([next_token])
                    put!(chan, token_str)
                    
                    last_token = next_token
                end
            end
            
        catch e
            if !(e isa InvalidStateException)
                @error "ERROR during CPU generation stream" exception=(e, catch_backtrace())
            end
        finally
            try
                close(chan)
            catch
            end
        end
    end
end

"""
    stream_to_stdout_cpu(model, prompt_tokens, decode_fn; kwargs...)

Generate tokens and print them to stdout as they are produced.

Returns the complete generated text as a String.
"""
function stream_to_stdout_cpu(model::QwenModelCPU, prompt_tokens::Vector{Int}, decode_fn::Function;
    max_tokens::Int=100,
    temperature::Float32=0.7f0,
    top_p::Float32=0.95f0,
    top_k::Int=20,
    repetition_penalty::Float32=1.0f0,
    stop_tokens::Set{Int}=Set{Int}(),
    io::IO=stdout)
    
    stream = generate_stream_cpu(model, prompt_tokens, decode_fn;
        max_tokens, temperature, top_p, top_k, repetition_penalty, stop_tokens)
    
    generated_text = IOBuffer()
    try
        for token in stream
            print(io, token)
            flush(io)
            print(generated_text, token)
        end
        println(io)
        flush(io)
        return String(take!(generated_text))
    catch e
        if e isa InterruptException
            println(io)
            flush(io)
            return String(take!(generated_text))
        else
            rethrow(e)
        end
    end
end

end # module
