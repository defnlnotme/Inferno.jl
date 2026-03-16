module Model

using oneAPI
using LinearAlgebra
using Statistics

export QwenConfig, QwenModel, KVCache, forward!, RMSNorm, MLP, GatedDeltaNet, FullAttention, DecoderLayer, init_kv_cache, free_kv_cache!, free_all_kv_caches!, free_model_gpu!

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

abstract type QuantMatrix end

struct IQ2XXSMatrix <: QuantMatrix
    data::Union{oneVector{UInt8}, Vector{UInt8}}
    K::Int
    N::Int
end

struct IQ2LookupTables{G, S, M}
    grid::G
    signs_table::S
    kmask::M
end

# Base.size for compatibility if needed
Base.size(m::IQ2XXSMatrix) = (m.N, m.K)
Base.size(m::IQ2XXSMatrix, d::Int) = d == 1 ? m.N : (d == 2 ? m.K : 1)

const IQ2XXS_GRID_GPU = Ref{oneVector{UInt64}}()
const KSIGNS_IQ2XS_GPU = Ref{oneVector{UInt8}}()
const KMASK_IQ2XS_GPU = Ref{oneVector{UInt8}}()

function init_gpu_tables(grid, signs_table, kmask)
    # Use DeviceBuffer but ensured by our fixed copyto! (GPU-driven)
    IQ2XXS_GRID_GPU[] = oneArray(grid)
    KSIGNS_IQ2XS_GPU[] = oneArray(signs_table)
    KMASK_IQ2XS_GPU[] = oneArray(kmask)
end

# --- Normalization ---
struct RMSNorm
    weight::oneVector{Float32}
    eps::Float32
end

# In-place fused RMSNorm + scale kernel (single GPU pass, no temporary arrays)
function rms_norm_kernel!(out, x, weight, eps, N)
    n = get_global_id(1)
    if n <= N
        # Each work-item normalises its own column
        # This is a simple per-column kernel; for seq>1 dispatch N=seq
        # We reuse `out` as scratch here; caller passes pre-allocated output
        # Actually we need to compute mean of squares per column: done via loop
        acc = 0.0f0
        @inbounds for i in 1:N
            v = x[i, n]
            acc += v * v
        end
        inv_rms = 1.0f0 / sqrt(acc / Float32(N) + eps)
        @inbounds for i in 1:N
            out[i, n] = x[i, n] * inv_rms * weight[i]
        end
    end
    return nothing
end

# Helper to allocate on SharedBuffer (RAM accessible to GPU) to avoid 458 bug and save VRAM
function oneSharedArray(T::Type, dims...)
    return oneArray{T, length(dims), oneAPI.oneL0.SharedBuffer}(undef, dims...)
end



# Corrected RMSNorm kernel to normalize over the first dimension (hidden size)
function rms_norm_kernel!(out, x, weight, eps, rows, cols)
    j = get_global_id(1)
    if j <= cols
        acc = 0.0f0
        @inbounds for i in 1:rows
            v = x[i, j]
            acc += v * v
        end
        inv_rms = 1.0f0 / sqrt(acc / Float32(rows) + eps)
        @inbounds for i in 1:rows
            out[i, j] = x[i, j] * inv_rms * weight[i]
        end
    end
    return nothing
end

function (norm::RMSNorm)(x::oneAPI.oneArray{Float32, N}) where N
    rows = size(x, 1)
    cols = length(x) ÷ rows
    out = similar(x)
    gs = min(cols, 256)
    gr = cld(cols, gs)
    @oneapi items=gs groups=gr rms_norm_kernel!(out, x, norm.weight, norm.eps, rows, cols)
    return out
end

function is_gpu(x)
    x isa oneAPI.oneArray && return true
    try
        p = parent(x)
        p === x && return false
        return is_gpu(p)
    catch
        return false
    end
end

function (norm::RMSNorm)(x::AbstractArray{Float32, N}) where N
    if is_gpu(x)
        inner = oneArray(x)
        return norm(inner)
    else
        # Real CPU fallback
        w_cpu = collect(norm.weight)
        m = sum(x .* x, dims=1) .* (1.0f0 / Float32(size(x, 1)))
        inv_rms = 1.0f0 ./ sqrt.(m .+ norm.eps)
        return x .* inv_rms .* w_cpu
    end
end

# In-place RMSNorm that writes into a pre-allocated buffer
function rmsnorm!(out::oneMatrix{Float32}, x::oneMatrix{Float32}, norm::RMSNorm)
    rows, cols = size(x)
    gs = min(cols, 256)
    gr = cld(cols, gs)
    @oneapi items=gs groups=gr rms_norm_kernel!(out, x, norm.weight, norm.eps, rows, cols)
    return out
end

function rmsnorm(x::oneMatrix{Float32}, norm::RMSNorm)
    out = similar(x)
    return rmsnorm!(out, x, norm)
end

# --- GPU Sampling Kernels ---

# Temperature scaling kernel
function temperature_scale_kernel!(scaled_logits, logits, inv_temp, N)
    i = get_global_id(1)
    if i <= N
        scaled_logits[i] = logits[i] * inv_temp
    end
    return
end

# --- Optimized Matrix Multiplication with Tiling ---

# Performance constants for 2-bit quantization
const IQ2_TILE_SIZE = 16  # Optimal for 2-bit data (16x16 tiles)
const IQ2_VEC_TILE = 32   # Vector operations tile size
const CACHE_LINE_SIZE = 64 # Cache line alignment

struct MatMulConfig
    N::Int
    M::Int
    K::Int
    tile_size::Int
end

# Optimized tiled matrix multiplication kernel (without shared memory for oneAPI compatibility)
function tiled_mat_mul_kernel!(res, A, B, config::MatMulConfig)
    # Each work item computes one element of the output
    i = get_global_id(1)
    j = get_global_id(2)
    
    if i <= config.N && j <= config.M
        val = 0.0f0
        
        # Process in tiles for better cache utilization
        num_tiles = cld(config.K, config.tile_size)
        for t in 1:num_tiles
            tile_start = (t-1) * config.tile_size
            tile_end = min(tile_start + config.tile_size, config.K)
            
            # Unrolled loop for better performance
            k = tile_start
            while k + 3 < tile_end
                val += A[i, k+1] * B[k+1, j]
                val += A[i, k+2] * B[k+2, j]
                val += A[i, k+3] * B[k+3, j]
                val += A[i, k+4] * B[k+4, j]
                k += 4
            end
            
            # Handle remaining elements
            while k < tile_end
                val += A[i, k+1] * B[k+1, j]
                k += 1
            end
        end
        
        res[i, j] = val
    end
    
    return nothing
end

# Optimized matrix-vector multiplication with vectorization
function tiled_mat_vec_kernel!(res, weight, x, K, N, tile_size)
    i = get_global_id(1)
    if i <= N
        val = 0.0f0
        
        # Process in tiles for better cache utilization
        num_tiles = cld(K, tile_size)
        for t in 1:num_tiles
            tile_start = (t-1) * tile_size
            tile_end = min(tile_start + tile_size, K)
            
            # Unrolled loop for better performance
            k = tile_start
            while k + 3 < tile_end
                val += weight[i, k+1] * x[k+1]
                val += weight[i, k+2] * x[k+2]
                val += weight[i, k+3] * x[k+3]
                val += weight[i, k+4] * x[k+4]
                k += 4
            end
            
            # Handle remaining elements
            while k < tile_end
                val += weight[i, k+1] * x[k+1]
                k += 1
            end
        end
        
        res[i] = val
    end
    return nothing
end

function mat_mul(weight::AbstractArray{Float32,2}, x::AbstractArray{Float32,2})
    if is_gpu(weight) && is_gpu(x)
        N, K = size(weight)
        S = size(x, 2)
        res = oneArray{Float32}(undef, N, S)
        
        # Use optimized tiled kernel for matrix-matrix multiplication
        if S > 1
            M = S  # B is K x S, so M = S
            # Use smaller tiles for better cache utilization with dense operations
            tile_size = min(16, max(8, cld(N, 256)))  # Adaptive tile size
            gs_x = tile_size
            gs_y = tile_size
            gr_x = cld(N, tile_size)
            gr_y = cld(M, tile_size)
            config = MatMulConfig(N, M, K, tile_size)
            @oneapi items=(gs_x, gs_y) groups=(gr_x, gr_y) tiled_mat_mul_kernel!(res, weight, x, config)
        else
            # Use optimized vector kernel for matrix-vector multiplication
            # Larger tile for vector operations to maximize throughput
            tile_size = min(IQ2_VEC_TILE, max(32, cld(N, 8)))  # Adaptive vector tile
            gs = min(N, 256)
            gr = cld(N, gs)
            @oneapi items=gs groups=gr tiled_mat_vec_kernel!(res, weight, x, K, N, tile_size)
        end
        
        return res
    else
        return Float32.(collect(weight) * collect(x))
    end
end

function mat_mul!(res::oneMatrix{Float32}, weight::oneMatrix{Float32}, x::oneMatrix{Float32})
    N, K = size(weight)
    S = size(x, 2)
    
    # Use optimized tiled kernels with adaptive sizing
    if S > 1
        M = S
        tile_size = min(16, max(8, cld(N, 256)))  # Adaptive tile size
        gs_x = tile_size
        gs_y = tile_size
        gr_x = cld(N, tile_size)
        gr_y = cld(M, tile_size)
        config = MatMulConfig(N, M, K, tile_size)
        @oneapi items=(gs_x, gs_y) groups=(gr_x, gr_y) tiled_mat_mul_kernel!(res, weight, x, config)
    else
        tile_size = min(IQ2_VEC_TILE, max(32, cld(N, 8)))  # Adaptive vector tile
        gs = min(N, 256)
        gr = cld(N, gs)
        @oneapi items=gs groups=gr tiled_mat_vec_kernel!(res, weight, x, K, N, tile_size)
    end
    
    return res
end

function mat_mul(weight::oneMatrix{Float32}, x::oneMatrix{Float32})
    N, K = size(weight)
    S = size(x, 2)
    if S == 1
        res = oneArray{Float32}(undef, N, 1)
        return mat_mul!(res, weight, x)
    else
        res = oneArray(zeros(Float32, N, S))
        return mat_mul!(res, weight, x)
    end
end

# Optimized tiled IQ2XXS matrix multiplication kernel
# Optimized for 2-bit quantization with proper tiling and memory coalescing
function tiled_mat_mul_iq2_xxs_kernel!(res, data, x, K, N, tables)
    # Use 2D work groups for better parallelism
    grid = tables.grid
    signs_table = tables.signs_table
    kmask = tables.kmask

    work_item_x = Int(get_global_id(1))  # Output row
    work_item_y = Int(get_global_id(2))  # Tile column
    
    # Tile dimensions optimized for 2-bit data
    tile_rows = IQ2_TILE_SIZE
    tile_cols = IQ2_TILE_SIZE
    
    # Calculate which output element this work item computes
    output_row = work_item_x
    if output_row <= N
        val = 0.0f0
        nb = K ÷ 256
        
        # Process in tiles for better cache utilization
        tiles_per_block = cld(nb, tile_cols)
        for tile_idx in 1:tiles_per_block
            block_start = (tile_idx - 1) * tile_cols
            block_end = min(block_start + tile_cols, nb)
            
            # Process each block within the tile
            for i in (block_start + 1):block_end
                base = (output_row - 1) * nb * 66 + (i - 1) * 66
                d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
                d = Float32(reinterpret(Float16, d_raw))
                
                # Process 8 sub-blocks with unrolled loops
                for ib32 in 0:7
                    qs_base = base + 2 + ib32 * 8
                    
                    # Reading 8 bytes as 2 UInt32 with aligned access
                    aux32_1 = (UInt32(data[qs_base+1])) | (UInt32(data[qs_base+2]) << 8) | 
                              (UInt32(data[qs_base+3]) << 16) | (UInt32(data[qs_base+4]) << 24)
                    aux32_2 = (UInt32(data[qs_base+5])) | (UInt32(data[qs_base+6]) << 8) | 
                              (UInt32(data[qs_base+7]) << 16) | (UInt32(data[qs_base+8]) << 24)
                    
                    db = d * (0.5f0 + Float32(aux32_2 >> 28)) * 0.25f0
                    
                    # Unrolled inner loops for better performance
                    for l in 0:3
                        grid_idx = Int((aux32_1 >> (8*l)) & 255) + 1
                        grid_val = grid[grid_idx]
                        
                        signs_idx = Int((aux32_2 >> (7*l)) & 127) + 1
                        signs = signs_table[signs_idx]
                        
                        # Process 8 elements with unrolled loop
                        x_base = (i-1)*256 + ib32*32 + l*8
                        for j in 0:7
                            byte_val = Int8((grid_val >> (8 * j)) & 255) - 8
                            is_neg = (signs & kmask[j+1]) != 0
                            f_w = db * Float32(byte_val) * (is_neg ? -1.0f0 : 1.0f0)
                            
                            x_idx = x_base + j + 1
                            val += f_w * x[x_idx]
                        end
                    end
                end
            end
        end
        
        res[output_row] = val
    end
    
    return nothing
end

# Original kernel for backward compatibility
function mat_mul_iq2_xxs_kernel!(res, data, x, K, N, tables)
    n = Int(get_global_id(1))
    grid = tables.grid
    signs_table = tables.signs_table
    kmask = tables.kmask
    if n <= N
        val = 0.0f0
        nb = K ÷ 256
        for i in 1:nb
            base = (n - 1) * nb * 66 + (i - 1) * 66
            d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
            d = Float32(reinterpret(Float16, d_raw))
            
            for ib32 in 0:7
                qs_base = base + 2 + ib32 * 8
                
                # Reading 8 bytes as 2 UInt32
                aux32_1 = (UInt32(data[qs_base+1])) | (UInt32(data[qs_base+2]) << 8) | 
                          (UInt32(data[qs_base+3]) << 16) | (UInt32(data[qs_base+4]) << 24)
                aux32_2 = (UInt32(data[qs_base+5])) | (UInt32(data[qs_base+6]) << 8) | 
                          (UInt32(data[qs_base+7]) << 16) | (UInt32(data[qs_base+8]) << 24)
                
                db = d * (0.5f0 + Float32(aux32_2 >> 28)) * 0.25f0
                
                for l in 0:3
                    grid_idx = Int((aux32_1 >> (8*l)) & 255) + 1
                    grid_val = grid[grid_idx]
                    
                    signs_idx = Int((aux32_2 >> (7*l)) & 127) + 1
                    signs = signs_table[signs_idx]
                    
                    for j in 0:7
                        byte_val = Int8((grid_val >> (8 * j)) & 255) - 8
                        is_neg = (signs & kmask[j+1]) != 0
                        f_w = db * Float32(byte_val) * (is_neg ? -1.0f0 : 1.0f0)
                        
                        x_idx = (i-1)*256 + ib32*32 + l*8 + j + 1
                        val += f_w * x[x_idx]
                    end
                end
            end
        end
        res[n] = val
    end
    return nothing
end

function mat_mul!(res::oneMatrix{Float32}, weight::IQ2XXSMatrix, x::oneMatrix{Float32})
    N, K = weight.N, weight.K
    S = size(x, 2)
    
    # Convert CPU data to GPU if needed
    weight_data_gpu = weight.data isa Vector{UInt8} ? oneArray(weight.data) : weight.data
    
    tables = IQ2LookupTables(IQ2XXS_GRID_GPU[], KSIGNS_IQ2XS_GPU[], KMASK_IQ2XS_GPU[])

    # Use tiled kernel for better performance with 2-bit quantization
    if S == 1
        # Use optimized tiled kernel for single sequence
        gs_x = min(N, IQ2_TILE_SIZE)
        gs_y = 1  # Single column for vector operation
        gr_x = cld(N, gs_x)
        gr_y = 1
        
        @oneapi items=(gs_x, gs_y) groups=(gr_x, gr_y) tiled_mat_mul_iq2_xxs_kernel!(res, weight_data_gpu, x, K, N, tables)
    else
        # For batch processing, use tiled kernel per sequence
        for s in 1:S
            v = @view x[:, s]
            r = @view res[:, s]
            
            gs_x = min(N, IQ2_TILE_SIZE)
            gs_y = 1
            gr_x = cld(N, gs_x)
            gr_y = 1
            
            @oneapi items=(gs_x, gs_y) groups=(gr_x, gr_y) tiled_mat_mul_iq2_xxs_kernel!(r, weight_data_gpu, v, K, N, tables)
        end
    end
    return res
end

function mat_mul(weight::IQ2XXSMatrix, x::oneMatrix{Float32})
    N, K = weight.N, weight.K
    S = size(x, 2)
    if S == 1
        res = oneArray{Float32}(undef, N, 1)
    else
        res = oneArray{Float32}(undef, N, S)
    end
    return mat_mul!(res, weight, x)
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

function mat_mul_AT_B_kernel!(res, A, B, M, N)
    n = get_global_id(1)
    if n <= N
        val = 0.0f0
        for m in 1:M
            @inbounds val += A[m, n] * B[m, 1]
        end
        res[n] = val
    end
    return nothing
end

function mat_mul_AT_B!(res, A, B)
    M, N = size(A) # M is head_dim, N is total_len
    gs = min(N, 256)
    gr = cld(N, gs)
    @oneapi items=gs groups=gr mat_mul_AT_B_kernel!(res, A, B, M, N)
    return res
end

function mat_mul_AB(A::AbstractArray{Float32,2}, B::AbstractArray{Float32,2})
    if is_gpu(A) && is_gpu(B)
        M, N = size(A)
        S = size(B, 2)
        res = oneArray{Float32}(undef, M, S)
        gs_x = min(M, 16)
        gs_y = min(S, 16)
        gr_x = cld(M, gs_x)
        gr_y = cld(S, gs_y)
        @oneapi items=(gs_x, gs_y) groups=(gr_x, gr_y) mat_mul_AB_kernel!(res, A, B, N, M, S)
        return res
    else
        return Float32.(collect(A) * collect(B))
    end
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
end

# --- GPU Softmax Kernel ---
# Stable softmax on a 1D slice of length `len` stored in `scores[1:len, 1]`
# Writes probabilities back into `probs[1:len, 1]`
function softmax_kernel!(probs, scores, len, scale)
    # Single work item does the whole softmax for one head (autoregressive decode)
    # Called with items=1, groups=1
    mx = -Inf32
    @inbounds for i in 1:len
        v = scores[i, 1] * scale
        if v > mx
            mx = v
        end
    end
    s = 0.0f0
    @inbounds for i in 1:len
        v = exp(scores[i, 1] * scale - mx)
        probs[i, 1] = v
        s += v
    end
    inv_s = 1.0f0 / s
    @inbounds for i in 1:len
        probs[i, 1] *= inv_s
    end
    return nothing
end

# --- Batched GPU Softmax for prefill ---
function batched_softmax_kernel!(probs, scores, total_len, n_heads, scale)
    # Each work item handles one head for the entire sequence
    h = get_global_id(1)
    if h <= n_heads
        mx = -Inf32
        @inbounds for i in 1:total_len
            v = scores[i, h] * scale
            if v > mx
                mx = v
            end
        end
        s = 0.0f0
        @inbounds for i in 1:total_len
            v = exp(scores[i, h] * scale - mx)
            probs[i, h] = v
            s += v
        end
        inv_s = 1.0f0 / s
        @inbounds for i in 1:total_len
            probs[i, h] *= inv_s
        end
    end
    return nothing
end

# --- GPU Kernels for SSM Operations ---

# GPU sigmoid kernel
function sigmoid_kernel!(out, x, N)
    i = get_global_id(1)
    if i <= N
        out[i] = 1.0f0 / (1.0f0 + exp(-x[i]))
    end
    return nothing
end

# GPU softplus kernel  
function softplus_kernel!(out, x, bias, N)
    i = get_global_id(1)
    if i <= N
        out[i] = log(1.0f0 + exp(x[i] + bias[i]))
    end
    return nothing
end

# GPU 1D convolution kernel for SSM (simplified for compatibility)
function conv1d_kernel!(out, input, weight, conv_state, kernel_size, channels, seq)
    c = get_global_id(1)
    if c <= channels
        for t in 1:seq
            # Update conv state ring buffer
            for k in 1:(kernel_size-1)
                conv_state[k, c] = conv_state[k+1, c]
            end
            conv_state[kernel_size, c] = input[c, t]
            
            # Compute convolution
            s = 0.0f0
            for k in 1:kernel_size
                s += weight[k, c] * conv_state[k, c]
            end
            out[c, t] = s
        end
    end
    return nothing
end

# GPU L2 normalization kernel for SSM
function l2_norm_ssm_kernel!(q, k, head_dim, num_heads, seq, eps)
    t = get_global_id(1)
    h = get_global_id(2)
    if t <= seq && h <= num_heads
        # Compute L2 norm for Q
        q_norm = 0.0f0
        for d in 1:head_dim
            q_norm += q[d, h, t] * q[d, h, t]
        end
        q_norm = sqrt(q_norm + eps)
        
        # Compute L2 norm for K
        k_norm = 0.0f0
        for d in 1:head_dim
            k_norm += k[d, h, t] * k[d, h, t]
        end
        k_norm = sqrt(k_norm + eps)
        
        # Normalize in-place
        for d in 1:head_dim
            q[d, h, t] /= q_norm
            k[d, h, t] /= k_norm
        end
    end
    return nothing
end

# --- SiLU ---
function silu(x::AbstractArray{Float32})
    return x .* (1.0f0 ./ (1.0f0 .+ exp.(-x)))
end

# --- Rotary Embedding (RoPE) ---
# --- Rotary Embedding (RoPE) ---
struct RotaryEmbedding
    dim::Int
    base::Float32
    inv_freq::Vector{Float32}
end

function RotaryEmbedding(dim::Int; base=10000000.0)
    inv_freq = 1.0f0 ./ (Float32(base) .^ (Float32.(range(0, stop=dim - 1, step=2)) ./ Float32(dim)))
    return RotaryEmbedding(dim, Float32(base), inv_freq)
end

function rope_kernel!(x, inv_freq, pos, d, h, seq, d_rope)
    idx = get_global_id(1)
    if idx <= (d_rope ÷ 2) * h * seq
        half_d = d_rope ÷ 2
        i = (idx - 1) % half_d + 1
        rem = (idx - 1) ÷ half_d
        head = rem % h + 1
        t = rem ÷ h + 1
        
        idx1 = 2*i - 1
        idx2 = 2*i
        
        p = Float32(pos + t - 1)
        freq = inv_freq[i] * p
        cos_val = cos(freq)
        sin_val = sin(freq)
        
        x1 = x[idx1, head, t]
        x2 = x[idx2, head, t]
        
        x[idx1, head, t] = x1 * cos_val - x2 * sin_val
        x[idx2, head, t] = x1 * sin_val + x2 * cos_val
    end
    return nothing
end

function (rope::RotaryEmbedding)(x::oneArray{Float32,3}, pos::Int)
    d, h, seq = size(x)
    d_rope = min(d, rope.dim)
    
    inv_freq_gpu = oneArray(rope.inv_freq)
    
    n_elements = (d_rope ÷ 2) * h * seq
    gs = min(n_elements, 256)
    gr = cld(n_elements, gs)
    
    @oneapi items=gs groups=gr rope_kernel!(x, inv_freq_gpu, pos, d, h, seq, d_rope)
    
    return x
end

# --- KV Cache ---
mutable struct KVCache
    k::oneArray{Float32,3}  # (head_dim, n_kv, max_seq)
    v::oneArray{Float32,3}
    pos::Int
end

# Free GPU memory for a KVCache - call this on error/cleanup
function free_kv_cache!(cache::KVCache)
    try
        if isdefined(cache, :k) && cache.k !== nothing
            oneAPI.unsafe_free!(cache.k)
            cache.k = oneArray{Float32,3}(undef, 0, 0, 0)  # Replace with empty array
        end
    catch e
        @warn "Error freeing KV cache k: $e"
    end
    try
        if isdefined(cache, :v) && cache.v !== nothing
            oneAPI.unsafe_free!(cache.v)
            cache.v = oneArray{Float32,3}(undef, 0, 0, 0)
        end
    catch e
        @warn "Error freeing KV cache v: $e"
    end
    cache.pos = 0
    return nothing
end

# Free all KV caches in a vector
function free_all_kv_caches!(caches::Vector{KVCache})
    for cache in caches
        free_kv_cache!(cache)
    end
    return nothing
end

function init_kv_cache(head_dim, n_kv, max_seq)
    # Correct memory-efficient initialization using DeviceBuffer
    # Avoids CPU zeros() blowup and uses explicit fill! which is 458-safe.
    seq = min(max_seq, 512)
    try
        k = oneArray{Float32}(undef, head_dim, n_kv, seq)
        v = oneArray{Float32}(undef, head_dim, n_kv, seq)
        fill!(k, 0.0f0)
        fill!(v, 0.0f0)
        oneAPI.synchronize()
        return KVCache(k, v, 0)
    catch e
        @error "Failed to initialize KV cache: $e"
        rethrow(e)
    end
end

function update_kv_cache!(cache::KVCache, k::oneArray{Float32,3}, v::oneArray{Float32,3})
    seq = size(k, 3)
    pos = cache.pos
    dk = size(cache.k, 1)
    nk = size(cache.k, 2)
    current_max = size(cache.k, 3)
    
    # Simple bounds check - fail if we exceed cache size
    if pos + seq > current_max
        error("KV cache overflow: pos=$pos seq=$seq current_max=$current_max. Increase max_seq in config.")
    end
    
    @views cache.k[1:dk, 1:nk, pos+1:pos+seq] .= k
    @views cache.v[1:dk, 1:nk, pos+1:pos+seq] .= v
    cache.pos += seq
    return cache.k, cache.v
end

# --- MLP ---
struct MLP
    gate_weight::Union{oneMatrix{Float32}, QuantMatrix}
    up_weight::Union{oneMatrix{Float32}, QuantMatrix}
    down_weight::Union{oneMatrix{Float32}, QuantMatrix}
end

function (m::MLP)(x::oneMatrix{Float32})
    # silu(gate) * up
    g = mat_mul(m.gate_weight, x)
    u = mat_mul(m.up_weight, x)

    # SiLU: x * sigmoid(x) — fused in-place to avoid extra allocation
    @. g = g * (1.0f0 / (1.0f0 + exp(-g)))

    g .*= u
    return mat_mul(m.down_weight, g)
end

# --- Full Attention Layer ---
# Q weight packs both Q and gate interleaved: attn_q output is (head_dim*2*n_heads, seq)
# Q = output[0:head_dim, h, :], gate = output[head_dim:2*head_dim, h, :]
struct FullAttention
    wq::Union{oneMatrix{Float32}, QuantMatrix}
    wk::Union{oneMatrix{Float32}, QuantMatrix}
    wv::Union{oneMatrix{Float32}, QuantMatrix}
    wo::Union{oneMatrix{Float32}, QuantMatrix}
    q_norm::RMSNorm
    k_norm::RMSNorm
    n_heads::Int
    n_kv::Int
    head_dim::Int
    
    # Pre-allocated GPU buffers (no more CPU buffers)
    decode_q_full::oneMatrix{Float32}
    decode_k::oneMatrix{Float32}
    decode_v::oneMatrix{Float32}
    decode_combined::oneMatrix{Float32}
    decode_scores::oneMatrix{Float32}
    decode_pb::oneMatrix{Float32}
    decode_out_h::oneMatrix{Float32}
    decode_wo_buf::oneMatrix{Float32}
    # GPU buffers for prefill (dynamically sized)
    prefill_scores::oneMatrix{Float32}
    prefill_pb::oneMatrix{Float32}
end

function FullAttention(wq, wk, wv, wo, q_norm, k_norm, config::QwenConfig)
    hd = config.head_dim
    n_heads = size(wq, 1) ÷ (hd * 2)
    n_kv = size(wk, 1) ÷ hd

    decode_q_full = oneArray(zeros(Float32, hd * 2 * n_heads, 1))
    decode_k = oneArray(zeros(Float32, hd * n_kv, 1))
    decode_v = oneArray(zeros(Float32, hd * n_kv, 1))
    decode_combined = oneArray(zeros(Float32, hd * n_heads, 1))
    
    # Fixed buffer sizes - sufficient for 4096 context
    max_len = 4096
    decode_scores = oneArray(zeros(Float32, max_len, 1))
    decode_pb = oneArray(zeros(Float32, max_len, 1))
    decode_out_h = oneArray(zeros(Float32, hd, 1))
    wo_out_size = size(wo, 1)
    decode_wo_buf = oneArray(zeros(Float32, wo_out_size, 1))
    oneAPI.synchronize()
    
    # Prefill buffers - fixed size for 4096 context
    prefill_scores = oneArray(zeros(Float32, max_len, n_heads))
    prefill_pb = oneArray(zeros(Float32, max_len, n_heads))
    
    return FullAttention(wq, wk, wv, wo, q_norm, k_norm, n_heads, n_kv, hd,
        decode_q_full, decode_k, decode_v, decode_combined, decode_scores, decode_pb, decode_out_h,
        decode_wo_buf, prefill_scores, prefill_pb)
end

function (m::FullAttention)(x::oneArray{Float32,2}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    hd, seq = m.head_dim, size(x, 2)

    # 1. Packed Q+gate projection
    if seq == 1
        q_full = mat_mul!(m.decode_q_full, m.wq, x)
        q_3d = reshape(q_full, hd * 2, m.n_heads, seq)
        q_only = view(q_3d, 1:hd, :, :)
        gate_raw = view(q_3d, hd+1:2*hd, :, :)
        
        k = mat_mul!(m.decode_k, m.wk, x)
        v = mat_mul!(m.decode_v, m.wv, x)
    else
        q_full = mat_mul(m.wq, x)
        q_3d = reshape(q_full, hd * 2, m.n_heads, seq)
        q_only = view(q_3d, 1:hd, :, :)
        gate_raw = view(q_3d, hd+1:2*hd, :, :)
        
        k = mat_mul(m.wk, x)
        v = mat_mul(m.wv, x)
    end

    # 3. Apply Q, K normalization (GPU-native, no transfers)
    q_normed = m.q_norm(reshape(q_only, hd, :))
    k_normed = m.k_norm(reshape(k, hd, :))

    q_2d = reshape(q_normed, hd, m.n_heads, seq)
    k_2d = reshape(k_normed, hd, m.n_kv, seq)
    v_2d = reshape(v, hd, m.n_kv, seq)

    # 4. RoPE
    q_rope = rope(q_2d, pos)
    k_rope = rope(k_2d, pos)

    # 5. Gating Q (fused in-place SiLU)
    gate_silu = gate_raw .* (1.0f0 ./ (1.0f0 .+ exp.(-gate_raw)))
    q_gated = q_rope .* gate_silu

    # 6. KV Cache
    K_cache, V_cache = update_kv_cache!(cache, k_rope, v_2d)

    # 7. Attention
    total_len = cache.pos
    scale = 1.0f0 / sqrt(Float32(hd))

    if seq == 1
        # Fixed buffers - no growth needed
        q_final = reshape(q_gated, hd, m.n_heads, 1) # (hd, n_heads, 1)
        kv_per_q = m.n_heads ÷ m.n_kv
        
        for h in 1:m.n_heads
            kh = (h - 1) ÷ kv_per_q + 1
            
            sc_view = view(m.decode_scores, 1:total_len, :)
            K_view = view(K_cache, :, kh, 1:total_len)
            q_view = view(q_final, :, h, :)
            
            mat_mul_AT_B!(sc_view, K_view, q_view)
            
            # GPU softmax - no CPU transfers!
            sc_view .*= scale  # Apply scaling on GPU
            pb_view = view(m.decode_pb, 1:total_len, :)
            @oneapi items=1 groups=1 softmax_kernel!(pb_view, sc_view, total_len, 1.0f0)
            
            out_view = view(m.decode_combined, (h-1)*hd+1:h*hd, :)
            V_view = view(V_cache, :, kh, 1:total_len)
            gs_x = min(hd, 16)
            gs_y = 1
            gr_x = cld(hd, gs_x)
            @oneapi items=(gs_x, gs_y) groups=(gr_x, 1) mat_mul_AB_kernel!(out_view, V_view, pb_view, total_len, hd, 1)
        end
        combined = m.decode_combined
        return mat_mul!(m.decode_wo_buf, m.wo, combined)
    else
        # Prefill path - use batched GPU softmax
        q_final = reshape(q_gated, hd, m.n_heads, seq)
        kv_per_q = m.n_heads ÷ m.n_kv
        combined_all = oneArray(zeros(Float32, hd * m.n_heads, seq))
        
        # Fixed buffers - no growth needed
        
        for h in 1:m.n_heads
            kh = (h - 1) ÷ kv_per_q + 1
            
            # Compute all scores for this head at once
            scores_view = view(m.prefill_scores, 1:(pos+seq), h:h)
            for s in 1:seq
                K_v = reshape(view(K_cache, :, kh, 1:(pos+s)), hd, :)
                q_v = reshape(q_final[:, h, s], :, 1)
                
                # scores: (pos+s, 1) = K_v' * q_v
                scores_s = mat_mul_AB(transpose(K_v), q_v)
                scores_view[1:(pos+s), 1] .= vec(scores_s)
            end
            
            # Apply GPU softmax to all scores at once
            pb_view = view(m.prefill_pb, 1:(pos+seq), h:h)
            @oneapi items=1 groups=1 batched_softmax_kernel!(pb_view, scores_view, pos+seq, 1, scale)
            
            # Compute attention output for all timesteps
            for s in 1:seq
                V_v = reshape(view(V_cache, :, kh, 1:(pos+s)), hd, :)
                pb_s = view(pb_view, 1:(pos+s), :)
                out_s = mat_mul_AB(V_v, pb_s)
                combined_all[(h-1)*hd+1:h*hd, s] .= vec(out_s)
            end
        end
        combined = combined_all
        return mat_mul(m.wo, combined)
    end
end

# --- Gated Delta Net (SSM Layer) ---
# Reference: qwen35.cpp build_layer_attn_linear
struct GatedDeltaNet
    in_proj::Union{oneMatrix{Float32}, QuantMatrix}     # wqkv: (hidden, 6144) — projects to qkv_mixed
    gate_proj::Union{oneMatrix{Float32}, QuantMatrix}   # wqkv_gate: (hidden, d_inner=2048) — projects to z
    ssm_out::Union{oneMatrix{Float32}, QuantMatrix}     # (d_inner, hidden)
    ssm_a::oneVector{Float32}       # (num_v_heads=16,) — log space decay
    ssm_alpha::Union{oneMatrix{Float32}, QuantMatrix}   # (hidden, num_v_heads=16) — dt projection
    ssm_beta::Union{oneMatrix{Float32}, QuantMatrix}    # (hidden, num_v_heads=16) — beta projection
    ssm_conv1d::oneArray{Float32,2} # (conv_kernel=4, conv_channels=6144) — F32
    ssm_conv1d_cpu::Matrix{Float32} # CPU copy of conv1d weights
    ssm_dt_bias::oneVector{Float32} # GPU dt bias
    ssm_norm::RMSNorm               # (head_v_dim=128,) for output norm
    # CPU state buffers (kept on CPU for efficiency)
    conv_state::Matrix{Float32} # CPU: (conv_kernel, conv_channels) — ring buffer  
    ssm_state::Array{Float32}    # CPU: (head_v_dim, head_k_dim, num_v_heads) state matrix
    # Dimensions
    num_v_heads::Int    # = 16 (ssm_time_step_rank)
    num_k_heads::Int    # = 16 (ssm_group_count)
    head_k_dim::Int     # = 128 (ssm_state_size)
    head_v_dim::Int     # = 128 (d_inner / num_v_heads)
    d_inner::Int        # = 2048
    # Essential GPU buffers for operations (decode path)
    decode_beta::oneMatrix{Float32}
    decode_alpha::oneMatrix{Float32}
    decode_decay_gate::oneMatrix{Float32}
    decode_conv_out::oneMatrix{Float32}
    decode_z::oneMatrix{Float32}
    decode_output_normed::oneMatrix{Float32}
    decode_gated::oneMatrix{Float32}
    # Prefill buffers
    prefill_beta::oneMatrix{Float32}
    prefill_alpha::oneMatrix{Float32}
end

function GatedDeltaNet(in_proj, gate_proj, ssm_out, ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_conv1d_cpu, ssm_dt_bias, ssm_norm,
                       conv_state, ssm_state, num_v_heads, num_k_heads, head_k_dim, head_v_dim, d_inner)
    conv_channels = size(ssm_conv1d, 2)
    
    # Keep essential GPU buffers only
    decode_beta = oneArray(zeros(Float32, num_v_heads, 1))
    decode_alpha = oneArray(zeros(Float32, num_v_heads, 1))
    decode_decay_gate = oneArray(zeros(Float32, num_v_heads, 1))
    decode_conv_out = oneArray(zeros(Float32, conv_channels, 1))
    decode_z = oneArray(zeros(Float32, d_inner, 1))
    decode_output_normed = oneArray(zeros(Float32, head_v_dim * num_v_heads, 1))
    decode_gated = oneArray(zeros(Float32, d_inner, 1))
    
    # Prefill buffers - sized for max context
    max_seq = 4096
    prefill_beta = oneArray(zeros(Float32, num_v_heads, max_seq))
    prefill_alpha = oneArray(zeros(Float32, num_v_heads, max_seq))
    oneAPI.synchronize()
    
    # Convert conv1d to GPU if needed
    ssm_conv1d_gpu = ssm_conv1d isa oneArray ? ssm_conv1d : oneArray(Float32.(ssm_conv1d))
    
    return GatedDeltaNet(in_proj, gate_proj, ssm_out, ssm_a, ssm_alpha, ssm_beta, ssm_conv1d_gpu, ssm_conv1d_cpu, ssm_dt_bias, ssm_norm,
                         conv_state, ssm_state, num_v_heads, num_k_heads, head_k_dim, head_v_dim, d_inner,
                         decode_beta, decode_alpha, decode_decay_gate, decode_conv_out,
                         decode_z, decode_output_normed, decode_gated,
                         prefill_beta, prefill_alpha)
end

function reset_states!(m::GatedDeltaNet)
    # Reset CPU state buffers
    fill!(m.conv_state, 0.0f0)
    fill!(m.ssm_state, 0.0f0)
end

function (m::GatedDeltaNet)(x::oneMatrix{Float32}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    hidden, seq = size(x)
    conv_channels = size(m.conv_state, 2)
    conv_kernel = 4
    
    if seq == 1
        # Decode path - optimized for minimal memory usage
        # 1. Input projections
        qkv_mixed = mat_mul(m.in_proj, x)    # (6144, 1)
        z         = mat_mul(m.gate_proj, x)  # (2048, 1)
        
        # 2. Beta and Alpha projections (GPU)
        beta_raw  = mat_mul(m.ssm_beta, x)    # (16, 1)
        alpha_raw = mat_mul(m.ssm_alpha, x)  # (16, 1)
        
        # 3. Compute Beta (sigmoid) and Alpha (softplus) on GPU
        gs = min(m.num_v_heads, 256)
        gr = cld(m.num_v_heads, gs)
        @oneapi items=gs groups=gr sigmoid_kernel!(m.decode_beta, beta_raw, m.num_v_heads)
        @oneapi items=gs groups=gr softplus_kernel!(m.decode_alpha, alpha_raw, m.ssm_dt_bias, m.num_v_heads)
        
        # 4. Compute decay gate on GPU and collect to CPU
        m.decode_decay_gate .= m.decode_alpha .* m.ssm_a
        decay_gate_cpu = collect(m.decode_decay_gate)
        beta_cpu = collect(m.decode_beta)
        z_cpu = collect(z)
        
        # 5. 1D convolution on CPU
        qkv_cpu = collect(qkv_mixed)
        
        # Shift conv state ring buffer - copy from next position
        for c in 1:conv_channels
            for k in 1:(conv_kernel-1)
                m.conv_state[k, c] = m.conv_state[k+1, c]
            end
            m.conv_state[conv_kernel, c] = qkv_cpu[c, 1]
        end
        
        # Compute convolution
        conv_out_cpu = zeros(Float32, conv_channels, 1)
        conv_w_cpu = m.ssm_conv1d_cpu
        for c in 1:conv_channels
            s = 0.0f0
            for k in 1:conv_kernel
                s += conv_w_cpu[k, c] * m.conv_state[k, c]
            end
            conv_out_cpu[c, 1] = s
        end
        
        # 6. SiLU on CPU
        conv_out_cpu .*= (1.0f0 ./ (1.0f0 .+ exp.(-conv_out_cpu)))
        
        # 7. Split into Q, K, V
        qkv_size = m.head_k_dim * m.num_k_heads  # 2048
        v_size = m.head_v_dim * m.num_v_heads     # 2048
        
        q_flat = @view conv_out_cpu[1:qkv_size, :]                          # (2048, 1)
        k_flat = @view conv_out_cpu[qkv_size+1:2*qkv_size, :]              # (2048, 1)
        v_flat = @view conv_out_cpu[2*qkv_size+1:2*qkv_size+v_size, :]     # (2048, 1)
        
        # 8. L2-normalize Q and K on CPU
        q_4d = reshape(q_flat, m.head_k_dim, m.num_k_heads, seq)
        k_4d = reshape(k_flat, m.head_k_dim, m.num_k_heads, seq)
        v_4d = reshape(v_flat, m.head_v_dim, m.num_v_heads, seq)
        
        for h in 1:m.num_k_heads
            q_vec = @view q_4d[:, h, 1]
            k_vec = @view k_4d[:, h, 1]
            q_norm_val = sqrt(sum(abs2, q_vec) + m.ssm_norm.eps)
            k_norm_val = sqrt(sum(abs2, k_vec) + m.ssm_norm.eps)
            q_4d[:, h, 1] ./= q_norm_val
            k_4d[:, h, 1] ./= k_norm_val
        end
        
        # 9. SSM recurrence on CPU - use in-place update of m.ssm_state
        for vh in 1:m.num_v_heads
            g = exp(decay_gate_cpu[vh, 1])
            b = beta_cpu[vh, 1]
            
            kh = vh
            k_vec = @view k_4d[:, kh, 1]
            v_vec = @view v_4d[:, vh, 1]
            q_vec = @view q_4d[:, kh, 1]
            
            # Update state in-place
            state = @view m.ssm_state[:, :, vh]
            for i in 1:m.head_v_dim
                vi_b = v_vec[i] * b
                for j in 1:m.head_k_dim
                    state[i, j] = g * state[i, j] + vi_b * k_vec[j]
                end
            end
        end
        
        # 10. Compute output on CPU
        output_cpu = zeros(Float32, m.head_v_dim * m.num_v_heads, seq)
        for vh in 1:m.num_v_heads
            kh = vh
            q_vec = @view q_4d[:, kh, 1]
            for i in 1:m.head_v_dim
                s = 0.0f0
                for j in 1:m.head_k_dim
                    s += m.ssm_state[i, j, vh] * q_vec[j]
                end
                output_cpu[(vh-1)*m.head_v_dim + i, 1] = s
            end
        end
        
        # 11. Final processing on GPU
        output_gpu = oneArray(output_cpu)
        rmsnorm!(m.decode_output_normed, output_gpu, m.ssm_norm)
        
        # GPU SiLU on z and gating
        z_silu = z_cpu .* (1.0f0 ./ (1.0f0 .+ exp.(-z_cpu)))
        z_silu_gpu = oneArray(z_silu)
        m.decode_gated .= z_silu_gpu
        m.decode_gated .*= m.decode_output_normed
        
        return mat_mul(m.ssm_out, m.decode_gated)
        
    else
        # Prefill path - simplified: compute on CPU to avoid complexity
        qkv_mixed = mat_mul(m.in_proj, x)
        z = mat_mul(m.gate_proj, x)
        
        # Compute beta and alpha on CPU
        beta_raw = collect(mat_mul(m.ssm_beta, x))
        alpha_raw = collect(mat_mul(m.ssm_alpha, x))
        
        seq = size(beta_raw, 2)
        
        # CPU sigmoid
        beta = 1.0f0 ./ (1.0f0 .+ exp.(-beta_raw))
        
        # CPU softplus: softplus(x) = log(1 + exp(x))
        ssm_dt_bias_cpu = collect(m.ssm_dt_bias)
        alpha_raw_cpu = alpha_raw .+ reshape(ssm_dt_bias_cpu, :, 1)
        alpha = log.(1.0f0 .+ exp.(alpha_raw_cpu))
        
        # Compute decay gate: alpha * ssm_a
        ssm_a_cpu = collect(m.ssm_a)
        decay_gate = alpha .* ssm_a_cpu
        
        # CPU computation for prefill
        qkv_cpu = collect(qkv_mixed)
        z_cpu = collect(z)
        beta_cpu = beta
        decay_gate_cpu = decay_gate
        
        # Process on CPU for prefill
        conv_out_cpu = zeros(Float32, conv_channels, seq)
        conv_state_cpu = zeros(Float32, conv_kernel, conv_channels)
        conv_w_cpu = m.ssm_conv1d_cpu
        
        for t in 1:seq
            # Update conv state
            for k in 1:(conv_kernel-1)
                conv_state_cpu[k, :] .= conv_state_cpu[k+1, :]
            end
            conv_state_cpu[conv_kernel, :] .= qkv_cpu[:, t]
            
            # Compute convolution
            for c in 1:conv_channels
                s = 0.0f0
                for k in 1:conv_kernel
                    s += conv_w_cpu[k, c] * conv_state_cpu[k, c]
                end
                conv_out_cpu[c, t] = s
            end
        end
        
        # SiLU
        conv_out_cpu .*= (1.0f0 ./ (1.0f0 .+ exp.(-conv_out_cpu)))
        
        # Split and process
        qkv_size = m.head_k_dim * m.num_k_heads
        v_size = m.head_v_dim * m.num_v_heads
        
        q_flat = @view conv_out_cpu[1:qkv_size, :]
        k_flat = @view conv_out_cpu[qkv_size+1:2*qkv_size, :]
        v_flat = @view conv_out_cpu[2*qkv_size+1:2*qkv_size+v_size, :]
        
        q_4d = reshape(q_flat, m.head_k_dim, m.num_k_heads, seq)
        k_4d = reshape(k_flat, m.head_k_dim, m.num_k_heads, seq)
        v_4d = reshape(v_flat, m.head_v_dim, m.num_v_heads, seq)
        
        # L2 normalization
        for t in 1:seq
            for h in 1:m.num_k_heads
                q_vec = @view q_4d[:, h, t]
                k_vec = @view k_4d[:, h, t]
                q_norm_val = sqrt(sum(abs2, q_vec) + m.ssm_norm.eps)
                k_norm_val = sqrt(sum(abs2, k_vec) + m.ssm_norm.eps)
                q_4d[:, h, t] ./= q_norm_val
                k_4d[:, h, t] ./= k_norm_val
            end
        end
        
        # SSM recurrence
        output_cpu = zeros(Float32, m.head_v_dim, m.num_v_heads, seq)
        ssm_state_cpu = zeros(Float32, m.head_v_dim, m.head_k_dim, m.num_v_heads)
        
        for t in 1:seq
            for vh in 1:m.num_v_heads
                g = exp(decay_gate_cpu[vh, t])
                b = beta_cpu[vh, t]
                
                kh = vh
                k_vec = @view k_4d[:, kh, t]
                v_vec = @view v_4d[:, vh, t]
                q_vec = @view q_4d[:, kh, t]
                
                state = @view ssm_state_cpu[:, :, vh]
                
                # Update state
                for i in 1:m.head_v_dim
                    vi_b = v_vec[i] * b
                    for j in 1:m.head_k_dim
                        state[i, j] = g * state[i, j] + vi_b * k_vec[j]
                    end
                end
                
                # Compute output
                for i in 1:m.head_v_dim
                    s = 0.0f0
                    for j in 1:m.head_k_dim
                        s += state[i, j] * q_vec[j]
                    end
                    output_cpu[i, vh, t] = s
                end
            end
        end
        
        # Final processing
        output_flat = reshape(output_cpu, m.head_v_dim * m.num_v_heads, seq)
        output_gpu = oneArray(output_flat)
        
        # Only use pre-allocated buffers if seq == 1 (decoding)
        output_normed = if seq == 1
            rmsnorm!(m.decode_output_normed, output_gpu, m.ssm_norm)
        else
            rmsnorm(output_gpu, m.ssm_norm)
        end
        
        z_silu = z_cpu .* (1.0f0 ./ (1.0f0 .+ exp.(-z_cpu)))
        z_silu_gpu = oneArray(z_silu)
        
        gated = output_normed .* z_silu_gpu
        
        return mat_mul(m.ssm_out, gated)
    end
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
    embed::Matrix{Float32} # CPU-based (kept for compatibility)
    layers::Vector{DecoderLayer}
    final_norm::RMSNorm
    lm_head::Matrix{Float32} # CPU-based (kept for compatibility)
    rope::RotaryEmbedding
end

# Free all GPU memory associated with a QwenModel (weights, norms, etc.)
# Note: This frees the GPU arrays but keeps the model structure for reloading
function free_model_gpu!(model::QwenModel)
    # Free embedding
    try
        if isdefined(model, :embed) && model.embed !== nothing
            # embed is CPU-based, no GPU memory to free
        end
    catch e
        @warn "Error freeing embed: $e"
    end
    
    # Free layers - each layer has attention/SSM and MLP with GPU arrays
    for layer in model.layers
        try
            # Free in_norm and post_norm
            if isdefined(layer, :in_norm) && isdefined(layer.in_norm, :weight)
                oneAPI.unsafe_free!(layer.in_norm.weight)
            end
            if isdefined(layer, :post_norm) && isdefined(layer.post_norm, :weight)
                oneAPI.unsafe_free!(layer.post_norm.weight)
            end
        catch e
            @warn "Error freeing layer norms: $e"
        end
        
        try
            # Free attention/SSM op
            if layer.op isa FullAttention
                # Free pre-allocated GPU buffers in FullAttention
                op = layer.op
                for field in fieldnames(FullAttention)
                    if field in [:decode_q_full, :decode_k, :decode_v, :decode_combined, 
                                 :decode_scores, :decode_pb, :decode_out_h, :decode_wo_buf,
                                 :prefill_scores, :prefill_pb]
                        try
                            buf = getfield(op, field)
                            if buf !== nothing
                                oneAPI.unsafe_free!(buf)
                            end
                        catch
                        end
                    end
                end
            elseif layer.op isa GatedDeltaNet
                # Free GPU buffers in GatedDeltaNet
                op = layer.op
                for field in fieldnames(GatedDeltaNet)
                    if field in [:decode_beta, :decode_alpha, :decode_decay_gate, 
                                 :decode_conv_out, :decode_z, :decode_output_normed,
                                 :decode_gated, :prefill_beta, :prefill_alpha]
                        try
                            buf = getfield(op, field)
                            if buf !== nothing
                                oneAPI.unsafe_free!(buf)
                            end
                        catch
                        end
                    end
                end
            end
        catch e
            @warn "Error freeing layer op: $e"
        end
        
        try
            # Free MLP
            mlp = layer.mlp
            for field in fieldnames(MLP)
                try
                    w = getfield(mlp, field)
                    if w !== nothing
                        oneAPI.unsafe_free!(w)
                    end
                catch
                end
            end
        catch e
            @warn "Error freeing MLP: $e"
        end
    end
    
    # Free final_norm
    try
        if isdefined(model, :final_norm) && isdefined(model.final_norm, :weight)
            oneAPI.unsafe_free!(model.final_norm.weight)
        end
    catch e
        @warn "Error freeing final_norm: $e"
    end
    
    # lm_head is CPU-based, no GPU memory to free
    
    return nothing
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
            if i % 6 == 0
                GC.gc(false)
            end
        end

        # 3. Final Norm — keep on GPU
        x_normed = model.final_norm(x)

        # 4. Final logits — collect normed tensor and do CPU BLAS
        x_final = collect(x_normed)

        # model.lm_head: (hidden, vocab) stored CPU-side
        # logits = lm_head' * x_final  → (vocab, seq)
        logits = (model.lm_head') * x_final

        return Float32.(logits)
    catch e
        println("ERROR in forward!: ", e)
        st = stacktrace(catch_backtrace())
        for line in st
            println("  ", line)
        end
        # Cleanup on error - free caches and force GC
        try
            free_all_kv_caches!(caches)
        catch
        end
        try
            GC.gc(true)
        catch
        end
        try
            oneAPI.synchronize()
        catch
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
