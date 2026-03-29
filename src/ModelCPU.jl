"""
"""CPU-only inference backend for Inferno.jl
This module provides pure CPU implementations without GPU dependencies.
"""
module ModelCPU

using LinearAlgebra
using Statistics
using ..QuantsCPU

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
    # MLA (Multi-Head Latent Attention for DeepSeek)
    q_lora_rank::Int = 0
    kv_lora_rank::Int = 0
    qk_rope_head_dim::Int = 0
    qk_nope_head_dim::Int = 0
    v_head_dim::Int = 0
end

# Helper functions
sigmoid(x) = 1.0f0 / (1.0f0 + exp(-x))

function sigmoid!(out::AbstractArray, x::AbstractArray)
    @. out = 1.0f0 / (1.0f0 + exp(-x))
    return out
end

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
end

function init_kv_cache_cpu(config::QwenConfigCPU, max_seq::Int = 4096)
    k = zeros(Float32, config.head_dim, config.num_key_value_heads, max_seq)
    v = zeros(Float32, config.head_dim, config.num_key_value_heads, max_seq)
    return KVCacheCPU(k, v)
end

function update_kv_cache!(cache::KVCacheCPU, k::Matrix{Float32}, v::Matrix{Float32}, pos::Int)
    @views cache.k[:, :, pos + 1] .= k
    @views cache.v[:, :, pos + 1] .= v
    return cache
end

# --- MLP ---
# Union type for weight matrices (either Float32 or quantized)
const QuantOrFloat32 = Union{Matrix{Float32}, Q4_K_Matrix, Q5_K_Matrix, Q6_K_Matrix, Q8_0_Matrix}

struct MLPCPU
    gate_weight::QuantOrFloat32 # (intermediate, hidden)
    up_weight::QuantOrFloat32
    down_weight::QuantOrFloat32
end

# MLP call for Float32 weights
function (mlp::MLPCPU)(x::Vector{Float32}) where {T<:Matrix{Float32}}
    # Gate with SiLU
    gate = mlp.gate_weight * x
    @. gate = gate * (1.0f0 / (1.0f0 + exp(-gate))) # SiLU
    
    # Up projection
    up = mlp.up_weight * x
    
    # Element-wise multiply
    hidden = gate .* up
    
    # Down projection
    return mlp.down_weight * hidden
end

# Helper: multiply quantized matrix by vector (row-wise)
function mul_quant_mat_vec(mat::Q4_K_Matrix, x::Vector{Float32}, out::Vector{Float32})
    # Dequantize and multiply row by row
    # mat is stored as (inner_dim, outer_dim), we need to compute mat' * x
    # which gives us a vector of size outer_dim
    fill!(out, 0.0f0)
    
    block_values = zeros(Float32, 256)
    
    for row in 1:mat.outer_dim
        sum_val = 0.0f0
        row_start = (row - 1) * mat.inner_dim
        
        for block in 0:(mat.inner_dim ÷ 256 - 1)
            global_block_idx = (row - 1) * (mat.inner_dim ÷ 256) + block
            block_offset = global_block_idx * QuantsCPU.Q4_K_BLOCK_SIZE + 1
            
            # Dequantize this block
            QuantsCPU.dequantize_q4_k_block!(block_values, mat.data, block_offset)
            
            for i in 1:256
                col_idx = block * 256 + i
                sum_val += block_values[i] * x[col_idx]
            end
        end
        out[row] = sum_val
    end
    return out
end

function mul_quant_mat_vec(mat::Q5_K_Matrix, x::Vector{Float32}, out::Vector{Float32})
    fill!(out, 0.0f0)
    block_values = zeros(Float32, 256)
    
    for row in 1:mat.outer_dim
        sum_val = 0.0f0
        
        for block in 0:(mat.inner_dim ÷ 256 - 1)
            global_block_idx = (row - 1) * (mat.inner_dim ÷ 256) + block
            block_offset = global_block_idx * QuantsCPU.Q5_K_BLOCK_SIZE + 1
            
            QuantsCPU.dequantize_q5_k_block!(block_values, mat.data, block_offset)
            
            for i in 1:256
                col_idx = block * 256 + i
                sum_val += block_values[i] * x[col_idx]
            end
        end
        out[row] = sum_val
    end
    return out
end

function mul_quant_mat_vec(mat::Q6_K_Matrix, x::Vector{Float32}, out::Vector{Float32})
    fill!(out, 0.0f0)
    block_values = zeros(Float32, 256)
    
    for row in 1:mat.outer_dim
        sum_val = 0.0f0
        
        for block in 0:(mat.inner_dim ÷ 256 - 1)
            global_block_idx = (row - 1) * (mat.inner_dim ÷ 256) + block
            block_offset = global_block_idx * QuantsCPU.Q6_K_BLOCK_SIZE + 1
            
            QuantsCPU.dequantize_q6_k_block!(block_values, mat.data, block_offset)
            
            for i in 1:256
                col_idx = block * 256 + i
                sum_val += block_values[i] * x[col_idx]
            end
        end
        out[row] = sum_val
    end
    return out
end

function mul_quant_mat_vec(mat::Q8_0_Matrix, x::Vector{Float32}, out::Vector{Float32})
    fill!(out, 0.0f0)
    block_values = zeros(Float32, 32)
    
    for row in 1:mat.outer_dim
        sum_val = 0.0f0
        
        for block in 0:(mat.inner_dim ÷ 32 - 1)
            global_block_idx = (row - 1) * (mat.inner_dim ÷ 32) + block
            block_offset = global_block_idx * QuantsCPU.Q8_0_BLOCK_SIZE + 1
            
            QuantsCPU.dequantize_q8_0_block!(block_values, mat.data, block_offset)
            
            for i in 1:32
                col_idx = block * 32 + i
                sum_val += block_values[i] * x[col_idx]
            end
        end
        out[row] = sum_val
    end
    return out
end

# Generic multiplication for quantized or Float32 weights
function mlp_mat_vec_mul(weight::Matrix{Float32}, x::Vector{Float32})
    return weight * x
end

function mlp_mat_vec_mul(weight::Q4_K_Matrix, x::Vector{Float32})
    out = Vector{Float32}(undef, weight.outer_dim)
    return mul_quant_mat_vec(weight, x, out)
end

function mlp_mat_vec_mul(weight::Q5_K_Matrix, x::Vector{Float32})
    out = Vector{Float32}(undef, weight.outer_dim)
    return mul_quant_mat_vec(weight, x, out)
end

function mlp_mat_vec_mul(weight::Q6_K_Matrix, x::Vector{Float32})
    out = Vector{Float32}(undef, weight.outer_dim)
    return mul_quant_mat_vec(weight, x, out)
end

function mlp_mat_vec_mul(weight::Q8_0_Matrix, x::Vector{Float32})
    out = Vector{Float32}(undef, weight.outer_dim)
    return mul_quant_mat_vec(weight, x, out)
end

# Generic MLP forward pass
function mlp_forward(mlp::MLPCPU, x::Vector{Float32})
    # Gate with SiLU
    gate = mlp_mat_vec_mul(mlp.gate_weight, x)
    @. gate = gate * (1.0f0 / (1.0f0 + exp(-gate))) # SiLU
    
    # Up projection
    up = mlp_mat_vec_mul(mlp.up_weight, x)
    
    # Element-wise multiply
    hidden = gate .* up
    
    # Down projection
    return mlp_mat_vec_mul(mlp.down_weight, hidden)
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
    
 # 3. Compute convolution - x_conv[c] = sum_k conv_state[c,k] * ssm_conv1d[k,c]
  # Using BLAS dot product for each channel (diagonal of matrix product)
  x_conv = Vector{Float32}(undef, m.conv_channels)
  for c in 1:m.conv_channels
      x_conv[c] = dot(view(m.conv_state, c, :), view(m.ssm_conv1d, :, c))
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
 # silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
 # Output: norm(y_all) * silu(z)
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
    
 # Apply Q/K normalization (per-head)
 # query_states has shape (head_dim, n_heads), normalize each column independently
 for h in 1:attn.n_heads
     q_h = view(query_states, :, h)
     rmsnorm_cpu!(q_h, q_h, attn.q_norm)
 end
 for h in 1:attn.n_kv
     k_h = view(k, :, h)
     rmsnorm_cpu!(k_h, k_h, attn.k_norm)
 end
    
    # Apply RoPE
    apply_rotary_emb!(query_states, pos, rope)
    apply_rotary_emb!(k, pos, rope)

    # Apply SiLU gating to Q before attention (matches GPU implementation)
    gate_silu = similar(gate)
    @. gate_silu = gate * (1.0f0 / (1.0f0 + exp(-gate)))  # SiLU
    gate_reshaped = reshape(gate_silu, attn.head_dim, attn.n_heads)
    query_states .*= gate_reshaped

    # Update KV cache
    update_kv_cache!(cache, k, v, pos)

    # Compute attention scores using BLAS
    # cache.k is (head_dim, n_kv, max_seq), cache.v is (head_dim, n_kv, max_seq)
    # query_states is (head_dim, n_heads)
    output = zeros(Float32, attn.n_heads * attn.head_dim)

    gqa_ratio = div(attn.n_heads, attn.n_kv)
    seq_len = pos + 1

    for h in 1:attn.n_heads
        kv_h = div(h - 1, gqa_ratio) + 1

        q_h = query_states[:, h]

        # Extract K and V for this KV head: (head_dim, seq_len)
        K_h = view(cache.k, :, kv_h, 1:seq_len)
        V_h = view(cache.v, :, kv_h, 1:seq_len)

        # Compute scores: K' * q = (seq_len, head_dim) * (head_dim,) = (seq_len,)
        # Using BLAS: scores = K_h' * q_h
        scores = K_h' * q_h
        scores .*= attn.scale

        # Softmax
        max_score = maximum(scores)
        scores = exp.(scores .- max_score)
        scores ./= sum(scores)

        # Weighted sum: V * scores = (head_dim, seq_len) * (seq_len,) = (head_dim,)
        out_h = V_h * scores

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

# --- Sampling Functions ---

function softmax_sample(logits::Vector{Float32}; temperature::Float32=1.0f0, top_p::Float32=1.0f0, top_k::Int=0, min_p::Float32=0.0f0)
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
        total = sum(probs)
        if total > 0.0f0
            probs ./= total
        end
    end
    
    # Apply minimum probability floor and renormalize
    if min_p > 0.0f0
        @inbounds for i in 1:length(probs)
            probs[i] = max(probs[i], min_p)
        end
        total = sum(probs)
        if total > 0.0f0
            probs ./= total
        end
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

function apply_presence_penalty!(logits::Vector{Float32}, token_counts::Dict{Int,Int}, penalty::Float32)
    if penalty == 0.0f0
        return
    end
    for (tokid, _cnt) in token_counts
        if 1 <= tokid <= length(logits)
            logits[tokid] -= penalty
        end
    end
end

function apply_repetition_penalty!(logits::Vector{Float32}, token_counts::Dict{Int,Int}, penalty::Float32)
    if penalty != 1.0f0
        for (tokid, _count) in token_counts
            if 1 <= tokid <= length(logits)
                if logits[tokid] > 0
                    logits[tokid] /= penalty
                else
                    logits[tokid] *= penalty
                end
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
    repetition_penalty::Float32=1.0f0, token_counts::Dict{Int,Int}=Dict{Int,Int}(), presence_penalty::Float32=0.0f0, min_p::Float32=0.0f0)
    
    # Forward pass
    logits = forward_cpu!(model, tokens, pos, caches)
    
    # Get logits for last token
    logits_vec = vec(logits[:, end])
    
    # Apply presence then repetition penalties
    apply_presence_penalty!(logits_vec, token_counts, presence_penalty)
    apply_repetition_penalty!(logits_vec, token_counts, repetition_penalty)
    
    # Sample
    next_token = softmax_sample(logits_vec; temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
    
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
    presence_penalty::Float32=0.0f0,
    min_p::Float32=0.0f0,
    stop_tokens::Set{Int}=Set{Int}())
    
    return Channel{String}(32) do chan
        try
            # Initialize caches and states using model's max position embeddings
            cache_size = model.config.max_position_embeddings
            caches = [init_kv_cache_cpu(model.config, cache_size) for _ in 1:model.config.num_hidden_layers]
            reset_states_cpu!(model)
            
            # Track token counts for repetition/presence penalty
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
                    temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, token_counts=token_counts, presence_penalty=presence_penalty, min_p=min_p)
                
                curr_pos += 1
                token_counts[next_token] = get(token_counts, next_token, 0) + 1
                
                # Decode and yield
                token_str = decode_fn([next_token])
                put!(chan, token_str)
                
                last_token = next_token
                
                # Generate remaining tokens
                for _ in 2:max_tokens
                    next_token, _ = generate_cpu(model, [last_token], curr_pos, caches;
                        temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, token_counts=token_counts, presence_penalty=presence_penalty, min_p=min_p)
                    
                    # Check stop token BEFORE updating state and yielding
                    if next_token in stop_tokens
                        break
                    end
                    
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
    presence_penalty::Float32=0.0f0,
    min_p::Float32=0.0f0,
    stop_tokens::Set{Int}=Set{Int}(),
    io::IO=stdout)
    
    stream = generate_stream_cpu(model, prompt_tokens, decode_fn;
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, min_p=min_p, stop_tokens=stop_tokens)
    
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

"""
    stream_to_stdout_cpu(model, tok, prompt; kwargs...)

Generate from a string prompt and print tokens as they are produced.
This is a convenience method that handles tokenization internally.

# Example
```julia
model, file = load_model_cpu("model.gguf")
tok = SimpleTokenizer(file)
stream_to_stdout_cpu(model, tok, "Hello, how are you?")
```
"""
function stream_to_stdout_cpu(model::QwenModelCPU, tok, prompt::String; kwargs...)
    prompt_tokens = encode_prompt(tok, prompt)
    decode_fn = (ids) -> decode_tokens(tok, ids)
    return stream_to_stdout_cpu(model, prompt_tokens, decode_fn; kwargs...)
end

# Helper functions for tokenization
function encode_prompt(tok, prompt::String)
 # Handle Vector{String} (raw token list from GGUF)
 if tok isa Vector{String}
 tokens_data = tok
 tokens = Int[]
 remaining = prompt
 is_first = true
 while !isempty(remaining)
 found = false
 for len in length(remaining):-1:1
 candidate = SubString(remaining, 1, len)
 # For first token, try without prefix first, then with prefix
 # For subsequent tokens, try with prefix first (space prefix in BPE)
 prefixes = is_first ? ["", "Ġ"] : ["Ġ", ""]
 for prefix in prefixes
 key = prefix * candidate
 idx = findfirst(==(key), tokens_data)
 if idx !== nothing
 push!(tokens, idx) # 1-indexed for Julia arrays
 remaining = len < length(remaining) ? SubString(remaining, len + 1) : ""
 found = true
 is_first = false
 break
 end
 end
 found && break
 end
 if !found
 remaining = length(remaining) > 1 ? SubString(remaining, 2) : ""
 end
 end
 return tokens
    # Handle BPETokenizer (already 1-indexed, pass through)
    elseif hasfield(typeof(tok), :id_to_token) && hasfield(typeof(tok), :merges)
        return Base.invokelatest(getfield(parentmodule(typeof(tok)), :encode), tok, prompt)
    # Handle SimpleTokenizer struct (0-indexed -> convert to 1-indexed)
    elseif hasfield(typeof(tok), :token_to_id)
        tokens = Int[]
        remaining = prompt
        while !isempty(remaining)
            found = false
            for len in length(remaining):-1:1
                candidate = SubString(remaining, 1, len)
                for prefix in ["", "Ġ"]
                    key = prefix * candidate
                    if haskey(tok.token_to_id, key)
                        push!(tokens, tok.token_to_id[key] + 1)  # Convert 0-indexed to 1-indexed
                        remaining = len < length(remaining) ? SubString(remaining, len + 1) : ""
                        found = true
                        break
                    end
                end
                found && break
            end
            if !found
                remaining = length(remaining) > 1 ? SubString(remaining, 2) : ""
            end
        end
        return tokens
    # Handle BPETokenizer (1-indexed encode -> convert to 0-indexed)
    else
        # Fallback: assume tok is a function
        return tok(prompt)
    end
end

function decode_tokens(tok, ids::Vector{Int})
    # Handle Vector{String} (raw token list from GGUF)
    if tok isa Vector{String}
        parts = String[]
        for id in ids
            if 1 <= id <= length(tok)
                t = tok[id]
                t = replace(t, "Ġ" => " ")
                push!(parts, t)
            end
        end
        return join(parts)
    # Handle SimpleTokenizer struct (0-indexed ids -> need +1 for 1-indexed tokens vector)
    elseif hasfield(typeof(tok), :tokens)
        parts = String[]
        for id in ids
            if 1 <= id <= length(tok.tokens)
                t = tok.tokens[id]
                t = replace(t, "Ġ" => " ")
                push!(parts, t)
            end
        end
        return join(parts)
    # Handle BPETokenizer (already 1-indexed, pass through)
    elseif hasfield(typeof(tok), :id_to_token) && hasfield(typeof(tok), :merges)
        return Base.invokelatest(getfield(parentmodule(typeof(tok)), :decode), tok, ids)
    else
        return tok(ids)
    end
end

end # module
