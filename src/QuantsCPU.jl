"""
CPU Quantized Weight Types and Operations

This module provides quantized weight storage for CPU inference, allowing
models to remain in their compressed quantized format rather than being
fully dequantized to Float32 at load time.

Block sizes for K-quants (all operate on 256-element blocks):
- Q4_K: 144 bytes/block (4.5 bits/element effective)
- Q5_K: 176 bytes/block (5.5 bits/element effective)
- Q6_K: 208 bytes/block (6.5 bits/element effective)
- Q8_0: 34 bytes/block (8 bits/element with block scaling)
"""
module QuantsCPU

using StaticArrays

export Q4_K_Matrix, Q5_K_Matrix, Q6_K_Matrix, Q8_0_Matrix
export dequantize_block!, dequantize_row!

# Block sizes for each quantization type
const Q4_K_BLOCK_SIZE = 144  # bytes per 256 elements
const Q5_K_BLOCK_SIZE = 176
const Q6_K_BLOCK_SIZE = 208
const Q8_0_BLOCK_SIZE = 34   # 32 elements per block

# ============================================================================
# Quantized Matrix Wrapper Types
# ============================================================================

"""
    Q4_K_Matrix

Q4_K quantized matrix stored in compressed format.
Data is stored as raw bytes, to be dequantized on-the-fly during computation.
"""
struct Q4_K_Matrix
    data::Vector{UInt8}      # Raw quantized data
    inner_dim::Int           # Inner dimension (e.g., hidden_size)
    outer_dim::Int           # Outer dimension (e.g., intermediate_size)
    num_blocks::Int          # Number of 256-element blocks
    
    function Q4_K_Matrix(data::Vector{UInt8}, inner_dim::Int, outer_dim::Int)
        num_elements = inner_dim * outer_dim
        @assert num_elements % 256 == 0 "Q4_K requires dimensions divisible by 256"
        num_blocks = num_elements ÷ 256
        expected_bytes = num_blocks * Q4_K_BLOCK_SIZE
        @assert length(data) >= expected_bytes "Insufficient data for Q4_K matrix"
        new(data, inner_dim, outer_dim, num_blocks)
    end
end

"""
    Q5_K_Matrix

Q5_K quantized matrix stored in compressed format.
"""
struct Q5_K_Matrix
    data::Vector{UInt8}
    inner_dim::Int
    outer_dim::Int
    num_blocks::Int
    
    function Q5_K_Matrix(data::Vector{UInt8}, inner_dim::Int, outer_dim::Int)
        num_elements = inner_dim * outer_dim
        @assert num_elements % 256 == 0 "Q5_K requires dimensions divisible by 256"
        num_blocks = num_elements ÷ 256
        expected_bytes = num_blocks * Q5_K_BLOCK_SIZE
        @assert length(data) >= expected_bytes "Insufficient data for Q5_K matrix"
        new(data, inner_dim, outer_dim, num_blocks)
    end
end

"""
    Q6_K_Matrix

Q6_K quantized matrix stored in compressed format.
"""
struct Q6_K_Matrix
    data::Vector{UInt8}
    inner_dim::Int
    outer_dim::Int
    num_blocks::Int
    
    function Q6_K_Matrix(data::Vector{UInt8}, inner_dim::Int, outer_dim::Int)
        num_elements = inner_dim * outer_dim
        @assert num_elements % 256 == 0 "Q6_K requires dimensions divisible by 256"
        num_blocks = num_elements ÷ 256
        expected_bytes = num_blocks * Q6_K_BLOCK_SIZE
        @assert length(data) >= expected_bytes "Insufficient data for Q6_K matrix"
        new(data, inner_dim, outer_dim, num_blocks)
    end
end

"""
    Q8_0_Matrix

Q8_0 quantized matrix stored in compressed format.
Block size is 32 elements (not 256 like K-quants).
"""
struct Q8_0_Matrix
    data::Vector{UInt8}
    inner_dim::Int
    outer_dim::Int
    num_blocks::Int
    
    function Q8_0_Matrix(data::Vector{UInt8}, inner_dim::Int, outer_dim::Int)
        num_elements = inner_dim * outer_dim
        @assert num_elements % 32 == 0 "Q8_0 requires dimensions divisible by 32"
        num_blocks = num_elements ÷ 32
        expected_bytes = num_blocks * Q8_0_BLOCK_SIZE
        @assert length(data) >= expected_bytes "Insufficient data for Q8_0 matrix"
        new(data, inner_dim, outer_dim, num_blocks)
    end
end

# ============================================================================
# Dequantization Functions
# ============================================================================

"""
    dequantize_q4_k_block(data::Vector{UInt8}, block_offset::Int) -> NTuple{256, Float32}

Dequantize a single Q4_K block (256 elements) starting at the given byte offset.
Returns a tuple of 256 Float32 values.

Q4_K block structure (144 bytes):
- d, dmin: 2*2 = 4 bytes (Float16 scales)
- scales: 12 bytes
- qs: 128 bytes (4-bit quantized values, 2 values per byte)
"""
function dequantize_q4_k_block(data::Vector{UInt8}, block_offset::Int)
    # Read scale values
    d = Float32(reinterpret(Float16, data[block_offset:block_offset+1])[1])
    dmin = Float32(reinterpret(Float16, data[block_offset+2:block_offset+3])[1])
    
    # Read scales (12 bytes)
    scales = @view data[block_offset+4:block_offset+15]
    
    # Read quantized data (128 bytes)
    qs = @view data[block_offset+16:block_offset+143]
    
    # Output array
    values = MVector{256, Float32}(undef)
    
    # Process 4 super-groups of 64 elements each
    for j in 0:3
        is_idx = 2 * j
        
        # sc1, m1 (first 32 elements of this super-group)
        sc1, m1 = if is_idx < 4
            UInt8(scales[is_idx + 1] & 63), UInt8(scales[is_idx + 5] & 63)
        else
            UInt8((scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4)),
            UInt8((scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4))
        end
        
        # sc2, m2 (next 32 elements of this super-group)
        sc2, m2 = if (is_idx + 1) < 4
            UInt8(scales[is_idx + 2] & 63), UInt8(scales[is_idx + 6] & 63)
        else
            UInt8((scales[is_idx + 6] & 0x0f) | ((scales[is_idx - 2] >> 6) << 4)),
            UInt8((scales[is_idx + 6] >> 4) | ((scales[is_idx + 2] >> 6) << 4))
        end
        
        d1, min1 = d * Float32(sc1), dmin * Float32(m1)
        d2, min2 = d * Float32(sc2), dmin * Float32(m2)
        
        base_idx = j * 64
        for l in 0:31
            ql_val = qs[j * 32 + l + 1]
            values[base_idx + l + 1] = d1 * Float32(ql_val & 0x0f) - min1
            values[base_idx + 32 + l + 1] = d2 * Float32(ql_val >> 4) - min2
        end
    end
    
    return NTuple{256, Float32}(values)
end

"""
    dequantize_q5_k_block(data::Vector{UInt8}, block_offset::Int) -> NTuple{256, Float32}

Dequantize a single Q5_K block (256 elements).

Q5_K block structure (176 bytes):
- d, dmin: 4 bytes (Float16)
- scales: 12 bytes
- qh: 32 bytes (high bits)
- qs: 128 bytes (low bits)
"""
function dequantize_q5_k_block(data::Vector{UInt8}, block_offset::Int)
    d = Float32(reinterpret(Float16, data[block_offset:block_offset+1])[1])
    dmin = Float32(reinterpret(Float16, data[block_offset+2:block_offset+3])[1])
    
    scales = @view data[block_offset+4:block_offset+15]
    qh = @view data[block_offset+16:block_offset+47]
    qs = @view data[block_offset+48:block_offset+175]
    
    values = MVector{256, Float32}(undef)
    
    for j in 0:3
        is_idx = 2 * j
        
        sc1, m1 = if is_idx < 4
            scales[is_idx + 1] & 63, scales[is_idx + 5] & 63
        else
            (scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4),
            (scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4)
        end
        
        sc2, m2 = if (is_idx + 1) < 4
            scales[is_idx + 2] & 63, scales[is_idx + 6] & 63
        else
            (scales[is_idx + 6] & 0x0f) | ((scales[is_idx - 2] >> 6) << 4),
            (scales[is_idx + 6] >> 4) | ((scales[is_idx + 2] >> 6) << 4)
        end
        
        d1, min1 = d * Float32(sc1), dmin * Float32(m1)
        d2, min2 = d * Float32(sc2), dmin * Float32(m2)
        
        base_idx = j * 64
        for l in 0:31
            qh_val = qh[j * 8 + (l ÷ 4) + 1]
            qs_val = qs[j * 32 + l + 1]
            
            # Combine low and high bits
            h_shift = (l % 4) * 2
            h_bit = (qh_val >> h_shift) & 0x03
            
            low1 = qs_val & 0x0f
            low2 = qs_val >> 4
            
            v1 = Float32((low1 | (h_bit << 4)) - 16)  # Range: -16 to 15
            v2 = Float32((low2 | ((qh_val >> (h_shift + 2)) << 4)) - 16)
            
            values[base_idx + l + 1] = d1 * v1 - min1
            values[base_idx + 32 + l + 1] = d2 * v2 - min2
        end
    end
    
    return NTuple{256, Float32}(values)
end

"""
    dequantize_q6_k_block(data::Vector{UInt8}, block_offset::Int) -> NTuple{256, Float32}

Dequantize a single Q6_K block (256 elements).

Q6_K block structure (208 bytes):
- ql: 128 bytes (low bits)
- qh: 64 bytes (high bits)  
- scales: 16 bytes
- d: 2 bytes (Float16)
"""
function dequantize_q6_k_block(data::Vector{UInt8}, block_offset::Int)
    ql = @view data[block_offset:block_offset+127]
    qh = @view data[block_offset+128:block_offset+191]
    scales_data = @view data[block_offset+192:block_offset+207]
    d = Float32(reinterpret(Float16, data[block_offset+208:block_offset+209])[1])
    
    values = MVector{256, Float32}(undef)
    
    for j in 0:255
        # Q6_K uses a more complex bit packing
        # Each element is 6 bits, with interleaved low/high bits
        block_idx = j ÷ 128
        inner_idx = j % 128
        
        # Low 4 bits from ql
        l = Int(ql[inner_idx + 1])
        # High 2 bits from qh
        h = Int((qh[inner_idx + 1] >> (2 * (j % 128))) & 0x03)
        
        # Combine and sign-extend
        raw_val = (l & 0x0f) | (h << 4)
        
        # Sign extend from 6 bits
        if raw_val >= 32
            raw_val -= 64
        end
        
        # Scale
        scale_idx = j ÷ 16 + 1
        scale = Float32(scales_data[scale_idx])
        
        values[j + 1] = d * scale * Float32(raw_val)
    end
    
    return NTuple{256, Float32}(values)
end

"""
    dequantize_q8_0_block(data::Vector{UInt8}, block_offset::Int) -> NTuple{32, Float32}

Dequantize a single Q8_0 block (32 elements).

Q8_0 block structure (34 bytes):
- d: 2 bytes (Float16 scale)
- qs: 32 bytes (8-bit signed values)
"""
function dequantize_q8_0_block(data::Vector{UInt8}, block_offset::Int)
    d = Float32(reinterpret(Float16, data[block_offset:block_offset+1])[1])
    qs = @view data[block_offset+2:block_offset+33]
    
    values = MVector{32, Float32}(undef)
    
    for i in 0:31
        # Interpret as signed int8
        v = reinterpret(Int8, qs[i + 1])
        values[i + 1] = d * Float32(v)
    end
    
    return NTuple{32, Float32}(values)
end

# ============================================================================
# Full Matrix Dequantization
# ============================================================================

"""
    dequantize_full!(out::Vector{Float32}, mat::Q4_K_Matrix)

Dequantize an entire Q4_K matrix to Float32.
"""
function dequantize_full!(out::Vector{Float32}, mat::Q4_K_Matrix)
    for i in 0:(mat.num_blocks - 1)
        block_offset = i * Q4_K_BLOCK_SIZE + 1
        values = dequantize_q4_k_block(mat.data, block_offset)
        
        # Calculate position in output
        block_idx = i * 256
        for j in 1:256
            out[block_idx + j] = values[j]
        end
    end
    return out
end

function dequantize_full!(out::Vector{Float32}, mat::Q5_K_Matrix)
    for i in 0:(mat.num_blocks - 1)
        block_offset = i * Q5_K_BLOCK_SIZE + 1
        values = dequantize_q5_k_block(mat.data, block_offset)
        block_idx = i * 256
        for j in 1:256
            out[block_idx + j] = values[j]
        end
    end
    return out
end

function dequantize_full!(out::Vector{Float32}, mat::Q6_K_Matrix)
    for i in 0:(mat.num_blocks - 1)
        block_offset = i * Q6_K_BLOCK_SIZE + 1
        values = dequantize_q6_k_block(mat.data, block_offset)
        block_idx = i * 256
        for j in 1:256
            out[block_idx + j] = values[j]
        end
    end
    return out
end

function dequantize_full!(out::Vector{Float32}, mat::Q8_0_Matrix)
    for i in 0:(mat.num_blocks - 1)
        block_offset = i * Q8_0_BLOCK_SIZE + 1
        values = dequantize_q8_0_block(mat.data, block_offset)
        block_idx = i * 32
        for j in 1:32
            out[block_idx + j] = values[j]
        end
    end
    return out
end

"""
    dequantize_to_array(mat) -> Matrix{Float32}

Dequantize a quantized matrix to a full Float32 matrix.
"""
function dequantize_to_array(mat::Q4_K_Matrix)
    out = Vector{Float32}(undef, mat.inner_dim * mat.outer_dim)
    dequantize_full!(out, mat)
    return reshape(out, mat.inner_dim, mat.outer_dim)
end

function dequantize_to_array(mat::Q5_K_Matrix)
    out = Vector{Float32}(undef, mat.inner_dim * mat.outer_dim)
    dequantize_full!(out, mat)
    return reshape(out, mat.inner_dim, mat.outer_dim)
end

function dequantize_to_array(mat::Q6_K_Matrix)
    out = Vector{Float32}(undef, mat.inner_dim * mat.outer_dim)
    dequantize_full!(out, mat)
    return reshape(out, mat.inner_dim, mat.outer_dim)
end

function dequantize_to_array(mat::Q8_0_Matrix)
    out = Vector{Float32}(undef, mat.inner_dim * mat.outer_dim)
    dequantize_full!(out, mat)
    return reshape(out, mat.inner_dim, mat.outer_dim)
end

# ============================================================================
# Matrix-Vector Multiplication (Dequantize-on-the-fly)
# ============================================================================

"""
    mul_quant_vec!(out::Vector{Float32}, mat, x::Vector{Float32})

Multiply a quantized matrix by a vector, dequantizing blocks on-the-fly.
This is the key function for memory-efficient inference.

The matrix is assumed to be stored as (inner_dim, outer_dim) and we compute
out = mat' * x (transpose multiplication) for the MLP layers.
"""
function mul_quant_vec!(out::Vector{Float32}, mat::Q4_K_Matrix, x::Vector{Float32})
    # For MLP: weight is (intermediate, hidden), input is (hidden,)
    # We compute output = weight * input, where weight is NOT transposed
    # So we iterate over rows (intermediate dimension)
    
    fill!(out, 0.0f0)
    
    # Temporary buffer for one block's worth of dequantized values
    block_values = MVector{256, Float32}(undef)
    
    # Process each row
    for row in 1:mat.outer_dim
        sum_val = 0.0f0
        row_start = (row - 1) * mat.inner_dim
        
        # Process blocks within the row
        for block in 0:(mat.inner_dim ÷ 256 - 1)
            # Calculate block index in column-major order
            # Data is stored as (inner, outer) = (inner_dim, outer_dim)
            # Block at position (col_block*256, row-1) 
            global_block_idx = (row - 1) * (mat.inner_dim ÷ 256) + block
            block_offset = global_block_idx * Q4_K_BLOCK_SIZE + 1
            
            # Dequantize this block
            dequantize_q4_k_block!(block_values, mat.data, block_offset)
            
            # Compute partial dot product
            for i in 1:256
                col_idx = block * 256 + i
                sum_val += block_values[i] * x[col_idx]
            end
        end
        
        out[row] = sum_val
    end
    
    return out
end

# For Q4_K, implement a row-by-row dequantization approach
function dequantize_q4_k_block!(out::AbstractVector{Float32}, data::Vector{UInt8}, block_offset::Int)
    d = Float32(reinterpret(Float16, data[block_offset:block_offset+1])[1])
    dmin = Float32(reinterpret(Float16, data[block_offset+2:block_offset+3])[1])
    
    scales = @view data[block_offset+4:block_offset+15]
    qs = @view data[block_offset+16:block_offset+143]
    
    for j in 0:3
        is_idx = 2 * j
        
        sc1, m1 = if is_idx < 4
            UInt8(scales[is_idx + 1] & 63), UInt8(scales[is_idx + 5] & 63)
        else
            UInt8((scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4)),
            UInt8((scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4))
        end
        
        sc2, m2 = if (is_idx + 1) < 4
            UInt8(scales[is_idx + 2] & 63), UInt8(scales[is_idx + 6] & 63)
        else
            UInt8((scales[is_idx + 6] & 0x0f) | ((scales[is_idx - 2] >> 6) << 4)),
            UInt8((scales[is_idx + 6] >> 4) | ((scales[is_idx + 2] >> 6) << 4))
        end
        
        d1, min1 = d * Float32(sc1), dmin * Float32(m1)
        d2, min2 = d * Float32(sc2), dmin * Float32(m2)
        
        base_idx = j * 64
        for l in 0:31
            ql_val = qs[j * 32 + l + 1]
            out[base_idx + l + 1] = d1 * Float32(ql_val & 0x0f) - min1
            out[base_idx + 32 + l + 1] = d2 * Float32(ql_val >> 4) - min2
        end
    end
    
    return out
end

function dequantize_q5_k_block!(out::AbstractVector{Float32}, data::Vector{UInt8}, block_offset::Int)
    d = Float32(reinterpret(Float16, data[block_offset:block_offset+1])[1])
    dmin = Float32(reinterpret(Float16, data[block_offset+2:block_offset+3])[1])
    
    scales = @view data[block_offset+4:block_offset+15]
    qh = @view data[block_offset+16:block_offset+47]
    qs = @view data[block_offset+48:block_offset+175]
    
    for j in 0:3
        is_idx = 2 * j
        
        sc1, m1 = if is_idx < 4
            scales[is_idx + 1] & 63, scales[is_idx + 5] & 63
        else
            (scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4),
            (scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4)
        end
        
        sc2, m2 = if (is_idx + 1) < 4
            scales[is_idx + 2] & 63, scales[is_idx + 6] & 63
        else
            (scales[is_idx + 6] & 0x0f) | ((scales[is_idx - 2] >> 6) << 4),
            (scales[is_idx + 6] >> 4) | ((scales[is_idx + 2] >> 6) << 4)
        end
        
        d1, min1 = d * Float32(sc1), dmin * Float32(m1)
        d2, min2 = d * Float32(sc2), dmin * Float32(m2)
        
        base_idx = j * 64
        for l in 0:31
            qh_val = qh[j * 8 + (l ÷ 4) + 1]
            qs_val = qs[j * 32 + l + 1]
            
            h_shift = (l % 4) * 2
            h_bit = (qh_val >> h_shift) & 0x03
            
            low1 = qs_val & 0x0f
            low2 = qs_val >> 4
            
            v1 = Float32((low1 | (h_bit << 4)) - 16)
            v2 = Float32((low2 | ((qh_val >> (h_shift + 2)) << 4)) - 16)
            
            out[base_idx + l + 1] = d1 * v1 - min1
            out[base_idx + 32 + l + 1] = d2 * v2 - min2
        end
    end
    
    return out
end

function dequantize_q6_k_block!(out::AbstractVector{Float32}, data::Vector{UInt8}, block_offset::Int)
    ql = @view data[block_offset:block_offset+127]
    qh = @view data[block_offset+128:block_offset+191]
    scales_data = @view data[block_offset+192:block_offset+207]
    d = Float32(reinterpret(Float16, data[block_offset+208:block_offset+209])[1])
    
    for j in 0:255
        inner_idx = j % 128
        
        l = Int(ql[inner_idx + 1])
        h = Int((qh[inner_idx + 1] >> (2 * (j % 128))) & 0x03)
        
        raw_val = (l & 0x0f) | (h << 4)
        
        if raw_val >= 32
            raw_val -= 64
        end
        
        scale_idx = j ÷ 16 + 1
        scale = Float32(scales_data[scale_idx])
        
        out[j + 1] = d * scale * Float32(raw_val)
    end
    
    return out
end

function dequantize_q8_0_block!(out::AbstractVector{Float32}, data::Vector{UInt8}, block_offset::Int)
    d = Float32(reinterpret(Float16, data[block_offset:block_offset+1])[1])
    qs = @view data[block_offset+2:block_offset+33]
    
    for i in 0:31
        v = reinterpret(Int8, qs[i + 1])
        out[i + 1] = d * Float32(v)
    end
    
    return out
end

# Type alias for union of all quantized matrix types
const QuantMatrixCPU = Union{Q4_K_Matrix, Q5_K_Matrix, Q6_K_Matrix, Q8_0_Matrix}

end # module
