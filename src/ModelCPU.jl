"""
CPU-only inference backend for Inferno.jl
This module provides pure CPU implementations without GPU dependencies.
"""
module ModelCPU

using LinearAlgebra
using Statistics
using ..QuantsCPU
using Printf

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
    # Using sum(abs2, x) is generally faster and more stable in Julia for this size
    ss = sum(abs2, x)
    scale = 1.0f0 / sqrt(ss / length(x) + norm.eps)
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
 # Formula: inv_freq[i] = 1.0 / (theta ^ (2i / rotary_dim))
 inv_freq = Float32[1.0 / (theta ^ (2(i-1)/rotary_dim)) for i in 1:div(rotary_dim, 2)]
 return RotaryEmbeddingCPU(inv_freq, max_seq_len, rotary_dim)
end

function apply_rotary_emb!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU)
 head_dim, num_heads = size(x, 1), size(x, 2)
 half = div(rope.rotary_dim, 2)
 
 # Precompute cos/sin for this position (shared across all heads)
 # This avoids recomputing cos/sin for each head
 cos_vals = @inbounds [cos(rope.inv_freq[i] * pos) for i in 1:half]
 sin_vals = @inbounds [sin(rope.inv_freq[i] * pos) for i in 1:half]
 
 for h in 1:num_heads
 for i in 1:half
 @inbounds begin
 idx1 = i
 idx2 = i + half
 
 x1 = x[idx1, h]
 x2 = x[idx2, h]
 
 c = cos_vals[i]
 s = sin_vals[i]
 
 x[idx1, h] = x1 * c - x2 * s
 x[idx2, h] = x1 * s + x2 * c
 end
 end
 end
 return x
end

# Fused RMSNorm + RoPE for attention (reduces memory passes)
function rmsnorm_rotary!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU, norm::RMSNormCPU)
 head_dim, num_heads = size(x, 1), size(x, 2)
 half = div(rope.rotary_dim, 2)
 weight = norm.weight
 
 # Precompute cos/sin for this position (shared across all heads)
 cos_vals = @inbounds [cos(rope.inv_freq[i] * pos) for i in 1:half]
 sin_vals = @inbounds [sin(rope.inv_freq[i] * pos) for i in 1:half]
 
 for h in 1:num_heads
 # First: RMSNorm on this head
 sum_sq = 0.0f0
 for i in 1:head_dim
 @inbounds sum_sq += x[i, h] * x[i, h]
 end
 rms = sqrt(sum_sq / head_dim + norm.eps)
 inv_rms = 1.0f0 / rms
 
 # Apply RMSNorm and RoPE in one pass
 for i in 1:head_dim
 @inbounds x[i, h] = x[i, h] * inv_rms * weight[i]
 end
 
 # Now apply RoPE to the first rotary_dim dimensions
 for i in 1:half
 @inbounds begin
 idx1 = i
 idx2 = i + half
 
 x1 = x[idx1, h]
 x2 = x[idx2, h]
 
 c = cos_vals[i]
 s = sin_vals[i]
 
 x[idx1, h] = x1 * c - x2 * s
 x[idx2, h] = x1 * s + x2 * c
 end
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
 # Manual copy to avoid allocation
 for h in 1:size(k, 2)
 for d in 1:size(k, 1)
 cache.k[d, h, pos + 1] = k[d, h]
 end
 end
 for h in 1:size(v, 2)
 for d in 1:size(v, 1)
 cache.v[d, h, pos + 1] = v[d, h]
 end
 end
 return cache
end

# --- MLP ---
# Union type for weight matrices (either Float32 or quantized)
const QuantOrFloat32 = Union{Matrix{Float32}, Q4_K_Matrix, Q5_K_Matrix, Q6_K_Matrix, Q8_0_Matrix}

struct MLPCPU
 gate_weight::QuantOrFloat32 # (intermediate, hidden) after GGUF reshape+transpose
 up_weight::QuantOrFloat32 # (intermediate, hidden)
 down_weight::QuantOrFloat32 # (hidden, intermediate)
 # Pre-allocated work buffers
 gate_buf::Vector{Float32}  # intermediate_size
 up_buf::Vector{Float32}    # intermediate_size
 hidden_buf::Vector{Float32} # intermediate_size
 output_buf::Vector{Float32} # hidden_size (for down projection output)
end

# MLP call - uses generic forward that handles both Float32 and quantized weights
function (mlp::MLPCPU)(x::Vector{Float32})
 return mlp_forward(mlp, x)
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
 # Use pre-allocated buffers
 gate_buf = mlp.gate_buf
 up_buf = mlp.up_buf
 hidden_buf = mlp.hidden_buf
 output_buf = mlp.output_buf
 
 # Gate with SiLU - compute into gate_buf
 mul!(gate_buf, mlp.gate_weight, x)
 @. gate_buf = gate_buf * (1.0f0 / (1.0f0 + exp(-gate_buf))) # SiLU
 
 # Up projection - compute into up_buf
 mul!(up_buf, mlp.up_weight, x)
 
 # Element-wise multiply - compute into hidden_buf
 @. hidden_buf = gate_buf * up_buf
 
 # Down projection - compute into output_buf and return
 mul!(output_buf, mlp.down_weight, hidden_buf)
 return output_buf
end

# --- GatedDeltaNet (SSM) ---
struct GatedDeltaNetCPU
 index::Int
 
 # Weight matrices (can be Float32 or quantized)
 in_proj::QuantOrFloat32 # (hidden, conv_channels) — projects to Q+K+V
 gate_proj::QuantOrFloat32 # (hidden, d_inner) — SiLU gate
 ssm_out::QuantOrFloat32 # (d_inner, hidden) — output projection
 ssm_conv1d::Matrix{Float32} # (conv_kernel, conv_channels) — transpose for column access
 
 # Alpha/beta projections (GGUF (heads, hidden) → Julia (hidden, heads) after reshape)
 ssm_alpha_weight::Matrix{Float32} # (hidden, num_v_heads)
 ssm_beta_weight::Matrix{Float32} # (hidden, num_v_heads)
 
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
 h::Array{Float32,3} # Per-group states: (head_v_dim, head_k_dim, num_v_heads)
 
 # Pre-allocated work buffers to avoid GC
 x_conv_buf::Vector{Float32}
 y_all_buf::Vector{Float32}
 alpha_proj_buf::Vector{Float32}
 beta_proj_buf::Vector{Float32}
 # Pre-allocated mat-vec output buffers
 qkv_buf::Vector{Float32}      # in_proj * x output
 z_buf::Vector{Float32}        # gate_proj * x output
 out_buf::Vector{Float32}      # ssm_out * y_all output
 # Per-head loop buffers
 sk_buf::Vector{Float32}       # state * k output (head_v_dim)
 d_buf::Vector{Float32}        # beta * (v - sk) output (head_v_dim)
 y_h_buf::Vector{Float32}      # state * q output (head_v_dim)
 # Per-head normalization buffers
 q_norm_buf::Vector{Float32}
 k_norm_buf::Vector{Float32}
end

function reset_states_cpu!(m::GatedDeltaNetCPU)
 fill!(m.conv_state, 0.0f0)
 fill!(m.h, 0.0f0)
 return nothing
end

function (m::GatedDeltaNetCPU)(x::Vector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
 # 1. Input projections using pre-allocated buffers
 qkv = m.qkv_buf
 z = m.z_buf
 mul!(qkv, m.in_proj, x)
 mul!(z, m.gate_proj, x)
 
 # 2. Update conv state (ring buffer)
 if m.conv_kernel > 1
 m.conv_state[:, 1:(m.conv_kernel-1)] .= m.conv_state[:, 2:m.conv_kernel]
 end
 m.conv_state[:, m.conv_kernel] .= qkv
 
 # 3. Compute convolution with fused SiLU activation
 # x_conv[c] = silu(sum_k conv_state[c,k] * ssm_conv1d[c,k])
 # Fused: compute dot product and apply SiLU in one step
 x_conv = m.x_conv_buf
 for c in 1:m.conv_channels
 # Fused conv + silu: silu(x) = x / (1 + exp(-x))
 v = dot(view(m.conv_state, c, :), view(m.ssm_conv1d, c, :))
 x_conv[c] = v / (1.0f0 + exp(-v))
 end
 
 # 5. Split into Q, K, V
 qk_size = m.head_k_dim * m.num_k_heads
 q_all = reshape(view(x_conv, 1:qk_size), m.head_k_dim, m.num_k_heads)
 k_all = reshape(view(x_conv, qk_size+1:2*qk_size), m.head_k_dim, m.num_k_heads)
 v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+m.d_inner), m.head_v_dim, m.num_v_heads)
 
 # 6. Alpha/beta projections (use pre-allocated buffers)
 # ssm_alpha_weight is (hidden, num_v_heads), need transpose for mat-vec mul
 mul!(m.alpha_proj_buf, m.ssm_alpha_weight', x)
 mul!(m.beta_proj_buf, m.ssm_beta_weight', x)
 alpha_proj = m.alpha_proj_buf
 beta_proj = m.beta_proj_buf
 
# 7. Process each head (delta net) - per-group states
# Use pre-allocated buffer
y_all = m.y_all_buf
fill!(y_all, 0.0f0)

eps = 1.0f-6

for h in 1:m.num_v_heads
 g = ((h - 1) % m.num_k_heads) + 1

 qg = view(q_all, :, g)
 kg = view(k_all, :, g)
 vg = view(v_all, :, h)

 # Apply scale factor (llama.cpp uses 1/sqrt(head_k_dim))
 scale = 1.0f0 / sqrt(Float32(m.head_k_dim))

 # L2 normalize q and k with fused scaling (as per llama.cpp ggml_l2_norm)
 # Fused: norm + divide + scale in one pass
 # Use pre-allocated buffers to avoid allocations in loop
 q_normalized = m.q_norm_buf
 k_normalized = m.k_norm_buf
 
 # Compute squared sums for L2 norm
 q_sum_sq = 0.0f0
 k_sum_sq = 0.0f0
 for i in 1:m.head_k_dim
 q_sum_sq += qg[i] * qg[i]
 k_sum_sq += kg[i] * kg[i]
 end
 
 # Fused normalize + scale in one pass
 q_norm_val = 1.0f0 / (sqrt(q_sum_sq) + eps) * scale
 k_norm_val = 1.0f0 / (sqrt(k_sum_sq) + eps)
 for i in 1:m.head_k_dim
 q_normalized[i] = qg[i] * q_norm_val
 k_normalized[i] = kg[i] * k_norm_val
 end

 # Gate values
 alpha_val = clamp(Float64(alpha_proj[h]) + Float64(m.ssm_dt_bias[h]), -20.0, 20.0)
 softplus_alpha = log(1.0 + exp(alpha_val))
 # HF formula: g = ssm_a * softplus(a + dt_bias)
 # ssm_a is already -exp(A_log), so we just multiply
 decay = Float32(m.ssm_a[h] * softplus_alpha)
 # Note: decay should be negative (from -exp(A_log)), and state *= exp(decay) = state * exp(g)
 # But we want to apply exp(g) to the state, so:
 # state .*= exp(decay) where decay = g = ssm_a * softplus_alpha
 # Actually, HF computes g and then the update is: state = state * exp(g) + ...
 # So we need: decay_to_apply = exp(g) = exp(ssm_a * softplus_alpha)
 decay_to_apply = Float32(exp(decay))

 beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
 beta_gate = Float32(1.0 / (1.0 + exp(-beta_val)))

 # State operations (DeltaNet autoregressive)
 # state: [head_v_dim, head_k_dim] = [128, 128]
 state = view(m.h, :, :, h)
 
 # Decay: state *= exp(g) where g = ssm_a * softplus_alpha
 state .*= decay_to_apply
 
 # sk = state * k (matrix-vector multiply)
 sk = m.sk_buf
 mul!(sk, state, k_normalized)
 
 # d = beta * (v - sk)
 d = m.d_buf
 @. d = beta_gate * (vg - sk)
 
 # State update: S = S + d * k' (outer product) - use BLAS.ger! for rank-1 update
 # ger!(alpha, x, y, A) computes A = alpha*x*y' + A
 BLAS.ger!(1.0f0, d, k_normalized, state)
 
 # y = state * q
 y_h = m.y_h_buf
 mul!(y_h, state, q_normalized)
 
 yg = view(y_all, (h-1)*m.head_v_dim+1:h*m.head_v_dim)
 yg .= y_h
end
# End of loop

    
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
# Removed variance scaling as it caused looping/instability
# y_all .*= 1.0f0 / sqrt(Float32(m.head_v_dim))
 
 # 10. Output projection using pre-allocated buffer
 out = m.out_buf
 mul!(out, m.ssm_out, y_all)
 return out
end

# --- SSM Matrix-Vector Multiplication Helpers ---
# These functions handle both Float32 and quantized weight matrices

function ssm_mat_vec_mul(weight::Matrix{Float32}, x::Vector{Float32})
 return weight * x
end

function ssm_mat_vec_mul(weight::Q4_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function ssm_mat_vec_mul(weight::Q5_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function ssm_mat_vec_mul(weight::Q6_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function ssm_mat_vec_mul(weight::Q8_0_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

# --- Full Attention ---
struct FullAttentionCPU
 index::Int
 wq::Matrix{Float32} # (n_heads * head_dim * 2, hidden) — query + gate projection
 wk::Matrix{Float32} # (n_kv * head_dim, hidden)
 wv::Matrix{Float32} # (n_kv * head_dim, hidden)
 wo::Matrix{Float32} # (hidden, n_heads * head_dim) — output projection
 q_norm::RMSNormCPU
 k_norm::RMSNormCPU
 n_heads::Int
 n_kv::Int
 head_dim::Int
 scale::Float32
 # Pre-allocated work buffers
 qkv_buf::Vector{Float32}      # wq output (n_heads * head_dim * 2)
 k_buf::Vector{Float32}        # wk output (n_kv * head_dim)
 v_buf::Vector{Float32}        # wv output (n_kv * head_dim)
 query_states_buf::Vector{Float32}  # query after split (n_heads * head_dim)
 gate_buf::Vector{Float32}          # gate after split (n_heads * head_dim)
 output_buf::Vector{Float32}        # final output (n_heads * head_dim)
 scores_buf::Vector{Float32}        # attention scores (max_seq_len)
 wo_output_buf::Vector{Float32}     # wo output (hidden_size)
end

function (attn::FullAttentionCPU)(x::Vector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
 # Q, K, V projections using pre-allocated buffers
 mul!(attn.qkv_buf, attn.wq, x)
 mul!(attn.k_buf, attn.wk, x)
 mul!(attn.v_buf, attn.wv, x)

 # Split qkv into query and gate - manual loop to avoid allocation
 query_states = attn.query_states_buf
 gate = attn.gate_buf
 
 for h in 1:attn.n_heads
 base = (h - 1) * (2 * attn.head_dim)
 for i in 1:attn.head_dim
 query_states[(h-1)*attn.head_dim+i] = attn.qkv_buf[base+i]
 gate[(h-1)*attn.head_dim+i] = attn.qkv_buf[base+attn.head_dim+i]
 end
 end

 # Reshape to (head_dim, num_heads)
 query_states = reshape(query_states, attn.head_dim, attn.n_heads)
 k = reshape(attn.k_buf, attn.head_dim, attn.n_kv)
 v = reshape(attn.v_buf, attn.head_dim, attn.n_kv)

 # Fused RMSNorm + RoPE for Q and K (reduces memory passes)
 rmsnorm_rotary!(query_states, pos, rope, attn.q_norm)
 rmsnorm_rotary!(k, pos, rope, attn.k_norm)

 # Update KV cache
 update_kv_cache!(cache, k, v, pos)

 # Compute attention scores using BLAS
 # Use pre-allocated output buffer
 output = attn.output_buf
 fill!(output, 0.0f0)

 gqa_ratio = div(attn.n_heads, attn.n_kv)
 seq_len = pos + 1

 for h in 1:attn.n_heads
 kv_h = div(h - 1, gqa_ratio) + 1

 # Use view instead of slice to avoid allocation
 q_h = view(query_states, :, h)

 # Extract K and V for this KV head: (head_dim, seq_len)
 K_h = view(cache.k, :, kv_h, 1:seq_len)
 V_h = view(cache.v, :, kv_h, 1:seq_len)

 # Compute scores: K' * q = (seq_len, head_dim) * (head_dim,) = (seq_len,)
 scores = view(attn.scores_buf, 1:seq_len)
 # Manual mat-vec multiply to avoid allocation
 for i in 1:seq_len
 scores[i] = dot(view(K_h, :, i), q_h) * attn.scale
 end

 # Softmax (in-place on scores buffer) - manual to avoid broadcast allocation
 max_score = maximum(scores)
 for i in 1:seq_len
 scores[i] = exp(scores[i] - max_score)
 end
 sum_scores = sum(scores)
 for i in 1:seq_len
 scores[i] /= sum_scores
 end

 # Weighted sum: V * scores = (head_dim, seq_len) * (seq_len,) = (head_dim,)
 # Manual mat-vec to avoid allocation
 for i in 1:attn.head_dim
 s = 0.0f0
 for j in 1:seq_len
 s += V_h[i, j] * scores[j]
 end
 output[(h-1)*attn.head_dim+i] = s
 end
 end

 # Apply sigmoid gate to output (gate comes from Q projection)
 # Manual loop to avoid broadcast allocation
 for i in 1:length(gate)
 gate[i] = 1.0f0 / (1.0f0 + exp(-gate[i]))
 end
 for i in 1:length(output)
 output[i] *= gate[i]
 end

 # Output projection using pre-allocated buffer
 mul!(attn.wo_output_buf, attn.wo, output)
 return attn.wo_output_buf
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

function forward_cpu!(model::QwenModelCPU, tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU}; full_logits::Bool=false)
 seq_len = length(tokens)

 if full_logits
 all_logits = zeros(Float32, model.config.vocab_size, seq_len)
 for t in 1:seq_len
 tok = tokens[t] # Already 1-indexed from encode()
 curr_pos = pos + t - 1
 # Copy embedding to avoid modifying the embedding matrix
 x = model.embed[:, tok]
 for (i, layer) in enumerate(model.layers)
 x = layer(x, curr_pos, model.rope, caches[i])
 end
 x = model.final_norm(x)
 all_logits[:, t] = model.lm_head * x
 end
 return all_logits
 else
 # Only compute LM head for the last position
 last_logits = Vector{Float32}(undef, model.config.vocab_size)
 for t in 1:seq_len
 tok = tokens[t] # Already 1-indexed from encode()
 curr_pos = pos + t - 1
 # Copy embedding to avoid modifying the embedding matrix
 x = model.embed[:, tok]
 for (i, layer) in enumerate(model.layers)
 x = layer(x, curr_pos, model.rope, caches[i])
 end
 x = model.final_norm(x)
 if t == seq_len
 mul!(last_logits, model.lm_head, x)
 end
 end
 return reshape(last_logits, model.config.vocab_size, 1)
 end
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
 # Handle temperature=0 (greedy/argmax sampling)
 if temperature == 0.0f0
 return argmax(logits)
 end
 
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
    
    # Apply minimum probability threshold (relative to max probability)
    if min_p > 0.0f0
        max_prob = maximum(probs)
        threshold = max_prob * min_p
        for i in 1:length(probs)
            if probs[i] < threshold
                probs[i] = 0.0f0
            end
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
 stop_tokens::Set{Int}=Set{Int}(),
 max_context::Int=8192) # Maximum context length for KV cache
 
 return Channel{String}(32) do chan
 try
 # Initialize caches with reasonable max_seq to avoid OOM
 max_cache_seq = min(model.config.max_position_embeddings, max_context)
 caches = [init_kv_cache_cpu(model.config, max_cache_seq) for _ in 1:model.config.num_hidden_layers]
 reset_states_cpu!(model)
            
            # Track token counts for repetition/presence penalty
            token_counts = Dict{Int,Int}()
            for t in prompt_tokens
                token_counts[t] = get(token_counts, t, 0) + 1
            end
            
# Process prompt tokens
curr_pos = 0
if !isempty(prompt_tokens)
    # Process all prompt tokens together to properly update states
    # logits has shape (vocab_size, length(prompt_tokens))
    logits = forward_cpu!(model, prompt_tokens, 0, caches)
    curr_pos = length(prompt_tokens)

    # The logits for the first generated token are the logits of the last prompt token
    first_logits_vec = vec(logits[:, end])

    # Apply penalties to the first token's logits
    apply_presence_penalty!(first_logits_vec, token_counts, presence_penalty)
    apply_repetition_penalty!(first_logits_vec, token_counts, repetition_penalty)

    # Sample the first token
    next_token = softmax_sample(first_logits_vec; temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)

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
 show_tps::Bool=false,
 io::IO=stdout,
 max_context::Int=8192)
 
 stream = generate_stream_cpu(model, prompt_tokens, decode_fn;
 max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, 
 repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, 
 min_p=min_p, stop_tokens=stop_tokens, max_context=max_context)
    
    generated_text = IOBuffer()
    try
        if show_tps
            t0 = time()
            token_count = 0
        end
        for token in stream
            print(io, token)
            flush(io)
            print(generated_text, token)
            if show_tps
                token_count += 1
            end
        end
        println(io)
        flush(io)
        if show_tps
            elapsed = time() - t0
            tps = elapsed > 0 ? token_count / elapsed : 0.0
            @printf(io, "[t/s] %.2f tokens/s — %d tokens in %.3fs\n", tps, token_count, elapsed)
            flush(io)
        end
        return String(take!(generated_text))
    catch e
        if e isa InterruptException
            println(io)
            flush(io)
            if show_tps
                elapsed = time() - t0
                tps = elapsed > 0 ? token_count / elapsed : 0.0
                @printf(io, "[t/s] %.2f tokens/s — %d tokens in %.3fs\n", tps, token_count, elapsed)
                flush(io)
            end
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
