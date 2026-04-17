# --- Flash Attention Implementation (CPU Optimized) ---
# This is included inside ModelCPU module

"""
    flash_attention_cpu!(output, Q, cache_k, cache_v, kv_h, seq_len, scale, head_dim)

Memory-efficient attention that avoids materializing the full attention matrix.

Key optimizations:
1. Tiled computation - process attention in blocks to fit in cache
2. Online softmax - compute softmax incrementally without full materialization
3. Recomputation of attention scores from KV cache instead of storing

This is essentially Flash Attention-2/3 adapted for CPU.
"""
function flash_attention_cpu!(
 output::AbstractVector{Float32},
 Q::AbstractVector{Float32},
 cache_k::AbstractArray{Float32,3},
 cache_v::AbstractArray{Float32,3},
 kv_h::Int,
 seq_len::Int,
 scale::Float32,
 head_dim::Int
)
    fill!(output, 0.0f0)
    
    BLOCK_N = 64  # Key/Value block size
    
    # Online softmax state
    m = -Inf32
    l = 0.0f0
    
    # Process KV cache in blocks
    for j in 1:BLOCK_N:seq_len
        j_end = min(j + BLOCK_N - 1, seq_len)
        block_len = j_end - j + 1
        
        scores = zeros(Float32, block_len)
        
        for n in 1:block_len
            k_j = j + n - 1
            s = 0.0f0
            @simd for d in 1:head_dim
                s += Q[d] * cache_k[d, kv_h, k_j]
            end
            scores[n] = s * scale
        end
        
        m_new = max(m, maximum(scores))
        scale_factor = exp(m - m_new)
        
        if m_new > m
            output .*= scale_factor
            l *= scale_factor
        end
        
        for n in 1:block_len
            k_j = j + n - 1
            p = exp(scores[n] - m_new)
            l += p
            @simd for d in 1:head_dim
                output[d] += p * cache_v[d, kv_h, k_j]
            end
        end
        
        m = m_new
    end
    
    inv_l = 1.0f0 / l
    @simd for d in 1:head_dim
        output[d] *= inv_l
    end
end
