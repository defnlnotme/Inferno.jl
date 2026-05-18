# --- Flash Attention Implementation (CPU Optimized) ---
# This is included inside ModelCPU module

"""
    flash_attention_cpu!(output, Q, cache_k, cache_v, kv_h, seq_len, scale, head_dim, scores_buf)

Memory-efficient attention that avoids materializing the full attention matrix.

Key optimizations:
1. Tiled computation - process attention in blocks to fit in cache
2. Online softmax - compute softmax incrementally without full materialization
3. Recomputation of attention scores from KV cache instead of storing
4. Zero-allocation - uses pre-allocated scores buffer and @turbo SIMD

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
    head_dim::Int,
    scores_buf::AbstractVector{Float32}
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
        
        # Use pre-allocated buffer view (zero allocations)
        scores = view(scores_buf, 1:block_len)
        
        for n in 1:block_len
            k_j = j + n - 1
            s = 0.0f0
            # @turbo for SIMD-accelerated dot product
            # We use views to make it clear to @turbo that these are contiguous
            K_col = view(cache_k, :, kv_h, k_j)
            @turbo for d in 1:head_dim
                s += Q[d] * K_col[d]
            end
            scores[n] = s * scale
        end
        
        # Incremental max for online softmax
        m_new = m
        for n in 1:block_len
            m_new = max(m_new, scores[n])
        end

        scale_factor = exp(m - m_new)
        
        # Rescale output if max changed
        if m_new > m
            @turbo for d in 1:head_dim
                output[d] *= scale_factor
            end
            l *= scale_factor
        end
        
        # Update output with current block
        for n in 1:block_len
            k_j = j + n - 1
            p = exp(scores[n] - m_new)
            l += p
            V_col = view(cache_v, :, kv_h, k_j)
            @turbo for d in 1:head_dim
                output[d] += p * V_col[d]
            end
        end
        
        m = m_new
    end
    
    # Final normalization
    inv_l = 1.0f0 / l
    @turbo for d in 1:head_dim
        output[d] *= inv_l
    end
end
