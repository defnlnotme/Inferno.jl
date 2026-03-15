module Dequant

using ..QuantsData

export dequantize_iq2_xxs, dequantize_iq2_xs, dequantize_iq2_s, dequantize_iq3_xxs, dequantize_iq3_s,
       dequantize_q2_k, dequantize_q3_k, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0

# --- Dequantization Logic ---

# --- IQ2_XXS Dequantization ---

function dequantize_iq2_xxs(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq2_xxs: 2 bytes scale (f16), 64 bytes quants
    # 256 elements per block
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    for i in 1:nb
        base = (i-1) * 66
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # 8 sub-blocks of 32 elements each
        for ib32 in 0:7
            qs_base = base + 2 + ib32 * 8
            
            # Read 8 bytes as 2 UInt32s
            aux32_1 = (UInt32(data[qs_base+1])) | (UInt32(data[qs_base+2]) << 8) | 
                      (UInt32(data[qs_base+3]) << 16) | (UInt32(data[qs_base+4]) << 24)
            aux32_2 = (UInt32(data[qs_base+5])) | (UInt32(data[qs_base+6]) << 8) | 
                      (UInt32(data[qs_base+7]) << 16) | (UInt32(data[qs_base+8]) << 24)
            
            db = d * (0.5f0 + (aux32_2 >> 28)) * 0.25f0
            
            # aux8 is the first 4 bytes (aux32_1)
            for l in 0:3
                grid_idx = ((aux32_1 >> (8*l)) & 0xFF) + 1
                grid_val = IQ2XXS_GRID[grid_idx]
                
                signs_idx = ((aux32_2 >> (7*l)) & 127) + 1
                signs = KSIGNS_IQ2XS[signs_idx]
                
                for j in 0:7
                    byte_val = Int8((grid_val >> (8 * j)) & 0xFF) - 8
                    is_neg = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[(i-1)*256 + ib32*32 + l*8 + j + 1] = db * Float32(byte_val) * (is_neg ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

# --- IQ2_XS Dequantization ---

function dequantize_iq2_xs(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq2_xs: 2 bytes scale (f16), 64 bytes quants (uint16_t[32]), 8 bytes scales
    # Total = 74 bytes per 256 elements
    @assert num_elements % 256 == 0
    nb = num_elements ÷ 256
    y = Vector{Float32}(undef, num_elements)
    
    for i in 1:nb
        base = (i-1) * 74
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # 8 sub-blocks of 32
        for ib32 in 0:7
            db0 = d * (0.5f0 + (data[base + 66 + ib32 + 1] & 0x0F)) * 0.25f0
            db1 = d * (0.5f0 + (data[base + 66 + ib32 + 1] >> 4)) * 0.25f0
            
            for l in 0:3
                qs_idx = base + 2 + (ib32*4 + l)*2
                v = (UInt16(data[qs_idx+2]) << 8) | UInt16(data[qs_idx+1])
                
                grid_idx = (v & 511) + 1
                grid_val = IQ2XS_GRID[grid_idx]
                
                signs_idx = (v >> 9) + 1
                signs = KSIGNS_IQ2XS[signs_idx]
                
                db = (l < 2) ? db0 : db1
                
                for j in 0:7
                    byte_val = Int8((grid_val >> (8 * j)) & 0xFF) - 8
                    is_neg = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[(i-1)*256 + ib32*32 + l*8 + j + 1] = db * Float32(byte_val) * (is_neg ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

# --- IQ2_S Dequantization ---

function dequantize_iq2_s(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq2_s: 2 bytes scale (f16), 64 bytes qs, 8 bytes qh, 8 bytes scales
    # Total = 82 bytes per 256 elements
    @assert num_elements % 256 == 0
    nb = num_elements ÷ 256
    y = Vector{Float32}(undef, num_elements)
    
    for i in 1:nb
        base = (i-1) * 82
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # signs start at data[base + 64 + 2 + 1] ??? no, wait.
        # qs[64], qh[8], scales[8]
        qs = @view data[base + 3:base + 2 + 64]
        qh = @view data[base + 3 + 64:base + 2 + 64 + 8]
        scales = @view data[base + 3 + 64 + 8:base + 2 + 64 + 8 + 8]
        # wait, signs in iq2_s? ggml-quants.c says: const uint8_t * signs = qs + QK_K/8;
        # Wait, if qs is 64 bytes (QK_K/4), and signs is 32 bytes (QK_K/8)...
        # block_iq2_s struct: uint8_t qs[QK_K/4], uint8_t qh[QK_K/32], uint8_t scales[QK_K/32].
        # In dequantize_row_iq2_s: signs = qs + QK_K/8;
        # That means signs are the second half of qs?
        # QK_K/4 = 64. QK_K/8 = 32. So signs = qs[33:64].
        
        for ib32 in 0:7
            db0 = d * (0.5f0 + (scales[ib32+1] & 0x0F)) * 0.25f0
            db1 = d * (0.5f0 + (scales[ib32+1] >> 4)) * 0.25f0
            
            for l in 0:3
                # grid index: qs[l] | (qh[ib32] << (8-2*l) & 0x300)
                # wait, which qs? qs for this ib32?
                # ib32 loop in C: qs is fixed? No, qs is x[i].qs.
                # y[j] loop: y += 8. l loop: l < 4.
                # Grid idx calculation: grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                # signs calculation: signs[l] & kmask_iq2xs[j]
                # Wait, signs is qs + 256/8 = qs + 32. 
                
                # So if we are in ib32, which part of qs?
                # ib32 goes 0..7. Each ib32 consumes 4 'y' increments of 8 elements = 32 elements.
                # So in total 32 loops of l*8? No. 8 ib32 * 4 l = 32.
                # So qs index should be ib32*4 + l.
                
                q_idx = ib32*4 + l + 1
                grid_idx = Int(qs[q_idx]) | (Int(qh[ib32+1]) << (8 - 2*l) & 0x300) + 1
                grid_val = IQ2S_GRID[grid_idx]
                
                sign_val = qs[q_idx + 32] # signs = qs + 32
                db = (l < 2) ? db0 : db1
                
                for j in 0:7
                    byte_val = Int8((grid_val >> (8 * j)) & 0xFF) - 8
                    is_neg = (sign_val & KMASK_IQ2XS[j+1]) != 0
                    y[(i-1)*256 + ib32*32 + l*8 + j + 1] = db * Float32(byte_val) * (is_neg ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

# --- IQ3_XXS Dequantization ---

function dequantize_iq3_xxs(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq3_xxs: 2 bytes scale (f16), 96 bytes quants (3*QK_K/8)
    # Total = 98 bytes per 256 elements
    @assert num_elements % 256 == 0
    nb = num_elements ÷ 256
    y = Vector{Float32}(undef, num_elements)
    
    for i in 1:nb
        base = (i-1) * 98
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # qs: 64 bytes? No, qs is x[i].qs. 
        # scales_and_signs = qs + 256/4 = qs + 64.
        # QK_K/4 = 64. 64 + 32 (scales) = 96. Total 98. Correct.
        
        for ib32 in 0:7
            # scales_and_signs is at data[base + 2 + 64]
            # aux32 = memcpy from scales_and_signs + 4*ib32. 
            # wait, ib32 goes 0..7. 4*7 = 28. fits in 32 bytes scales.
            ss_base = base + 2 + 64 + 4*ib32
            aux32 = (UInt32(data[ss_base+1])) | (UInt32(data[ss_base+2]) << 8) | 
                    (UInt32(data[ss_base+3]) << 16) | (UInt32(data[ss_base+4]) << 24)
            
            db = d * (0.5f0 + (aux32 >> 28)) * 0.5f0
            
            for l in 0:3
                signs_idx = ((aux32 >> (7*l)) & 127) + 1
                signs = KSIGNS_IQ2XS[signs_idx]
                
                # qs indices for this ib32 and l
                # ib32*8 + 2*l
                # Wait, qs is at base + 2.
                # ib32 goes 0..7. l goes 0..3. Total 32 l-increments.
                # Each l increment consumes 2 qs values. Total 64 qs values. fits.
                q_base = base + 2 + ib32*8 + 2*l
                grid1 = IQ3XXS_GRID[data[q_base+1] + 1]
                grid2 = IQ3XXS_GRID[data[q_base+2] + 1]
                
                for j in 0:3
                    y_idx = (i-1)*256 + ib32*32 + l*8 + j + 1
                    byte_val1 = Int8((grid1 >> (8 * j)) & 0xFF) - 4
                    is_neg1 = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[y_idx] = db * Float32(byte_val1) * (is_neg1 ? -1.0f0 : 1.0f0)
                    
                    byte_val2 = Int8((grid2 >> (8 * j)) & 0xFF) - 4
                    is_neg2 = (signs & KMASK_IQ2XS[j+5]) != 0
                    y[y_idx + 4] = db * Float32(byte_val2) * (is_neg2 ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

# --- IQ3_S Dequantization ---

function dequantize_iq3_s(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq3_s: 2 bytes d, 64 bytes qs, 8 bytes qh, 32 bytes signs, 4 bytes scales
    # Total = 110 bytes per 256 elements
    @assert num_elements % 256 == 0
    nb = num_elements ÷ 256
    y = Vector{Float32}(undef, num_elements)
    
    for i in 1:nb
        base = (i-1) * 110
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # ib32 += 2 loop means we process 64 elements at a time
        for ib32 in 0:2:7
            scales_val = data[base + 2 + 64 + 8 + 32 + (ib32 ÷ 2) + 1]
            db1 = d * (1.0f0 + 2.0f0 * (scales_val & 0x0F))
            db2 = d * (1.0f0 + 2.0f0 * (scales_val >> 4))
            
            # First 32 elements (db1)
            for l in 0:3
                q_idx = ib32 * 8 + 2 * l + 1 # wait, ib32*8? 
                # C code: qs += 8 AFTER the first l loop.
                # So the first l loop uses qs indices 0..7.
                # The second l loop uses qs indices 8..15.
                # But ib32 advances by 2. So ib32=0 uses 0..15, ib32=2 uses 16..31?
                # QK_K/4 = 64 bytes for qs. 256 elements.
                # Total 32 l-increments for the whole block?
                # ib32 = 0, 2, 4, 6. For each ib32, two l loops of 4. Total 4 * 2 * 4 = 32 l loops. 32 * 8 = 256. Correct.
                
                # Global qs index for ib32 and l
                qs_off = ib32 * 8
                qh_val = data[base + 2 + 64 + (ib32) + 1]
                signs_off = base + 2 + 64 + 8 + ib32 * 4
                
                # First l loop (db1)
                q_base = base + 2 + qs_off + 2*l
                grid1 = IQ3S_GRID[(Int(data[q_base+1]) | (Int(qh_val << (8 - 2*l)) & 256)) + 1]
                grid2 = IQ3S_GRID[(Int(data[q_base+2]) | (Int(qh_val << (7 - 2*l)) & 256)) + 1]
                
                signs = data[signs_off + l + 1]
                
                for j in 0:3
                    y_idx = (i-1)*256 + ib32*32 + l*8 + j + 1
                    byte_val1 = Int8((grid1 >> (8 * j)) & 0xFF) - 1
                    is_neg1 = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[y_idx] = db1 * Float32(byte_val1) * (is_neg1 ? -1.0f0 : 1.0f0)
                    
                    byte_val2 = Int8((grid2 >> (8 * j)) & 0xFF) - 1
                    is_neg2 = (signs & KMASK_IQ2XS[j+5]) != 0
                    y[y_idx + 4] = db1 * Float32(byte_val2) * (is_neg2 ? -1.0f0 : 1.0f0)
                end
            end
            
            # Second 32 elements (db2) - ib32+1? 
            # In C: ib32 increment is 2. it uses scales[ib32/2] for both db1 and db2.
            # db1 = d * (1 + 2*(scales & 0xf)), db2 = d * (1 + 2*(scales >> 4))
            # The second l loop uses qs += 8.
            
            for l in 0:3
                qs_off = (ib32 + 1) * 8
                qh_val = data[base + 2 + 64 + (ib32 + 1) + 1]
                signs_off = base + 2 + 64 + 8 + (ib32 + 1) * 4
                
                q_base = base + 2 + qs_off + 2*l
                grid1 = IQ3S_GRID[(Int(data[q_base+1]) | (Int(qh_val << (8 - 2*l)) & 256)) + 1]
                grid2 = IQ3S_GRID[(Int(data[q_base+2]) | (Int(qh_val << (7 - 2*l)) & 256)) + 1]
                
                signs = data[signs_off + l + 1]
                
                for j in 0:3
                    y_idx = (i-1)*256 + (ib32+1)*32 + l*8 + j + 1
                    byte_val1 = Int8((grid1 >> (8 * j)) & 0xFF) - 1
                    is_neg1 = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[y_idx] = db2 * Float32(byte_val1) * (is_neg1 ? -1.0f0 : 1.0f0)
                    
                    byte_val2 = Int8((grid2 >> (8 * j)) & 0xFF) - 1
                    is_neg2 = (signs & KMASK_IQ2XS[j+5]) != 0
                    y[y_idx + 4] = db2 * Float32(byte_val2) * (is_neg2 ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

function dequantize_q8_0(data::AbstractVector{UInt8}, num_elements::Int)
    # block_q8_0: 2 bytes scale (f16), 32 bytes quants (int8)
    # 32 elements per block
    @assert num_elements % 32 == 0
    nb = num_elements ÷ 32
    y = Vector{Float32}(undef, num_elements)
    for i in 1:nb
        base = (i-1) * 34
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        for j in 1:32
            q_u8 = data[base+2+j]
            q = Float32(reinterpret(Int8, q_u8))
            y[(i-1)*32 + j] = d * q
        end
    end
    return y
end

# --- Q4_K Dequantization ---

function get_scale_min_k4!(j::Int, q::AbstractVector{UInt8}, d_ref::Ref{UInt8}, m_ref::Ref{UInt8})
    if j < 4
        d_ref[] = q[j+1] & 63
        m_ref[] = q[j+5] & 63
    else
        d_ref[] = (q[j+5] & 0xF) | ((q[j-3] >> 6) << 4)
        m_ref[] = (q[j+5] >> 4) | ((q[j+1] >> 6) << 4)
    end
end

function dequantize_q4_k(data::AbstractVector{UInt8}, num_elements::Int)
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    # block_q4_K: 
    #   d, dmin: 2*2 = 4 bytes
    #   scales: K_SCALE_SIZE = 12 bytes
    #   qs: QK_K/2 = 128 bytes
    # Total = 4 + 12 + 128 = 144 bytes per block.
    
    d_ref = Ref{UInt8}(0)
    m_ref = Ref{UInt8}(0)
    
    for i in 1:nb
        base = (i-1) * 144
        d_val = Float32(reinterpret(Float16, (UInt16(data[base+2]) << 8) | UInt16(data[base+1])))
        m_val = Float32(reinterpret(Float16, (UInt16(data[base+4]) << 8) | UInt16(data[base+3])))
        
        scales = @view data[base+5:base+16]
        qs = @view data[base+17:base+144]
        
        is_idx = 0
        q_idx = 0
        for j in 0:64:255
            get_scale_min_k4!(is_idx, scales, d_ref, m_ref)
            d1 = d_val * d_ref[]
            m1 = m_val * m_ref[]
            
            get_scale_min_k4!(is_idx + 1, scales, d_ref, m_ref)
            d2 = d_val * d_ref[]
            m2 = m_val * m_ref[]
            
            for l in 0:31
                q_val = qs[q_idx + l + 1]
                y[(i-1)*256 + j + l + 1] = d1 * (q_val & 0xF) - m1
                y[(i-1)*256 + j + l + 33] = d2 * (q_val >> 4) - m2
            end
            q_idx += 32
            is_idx += 2
        end
    end
    return y
end

# --- Q5_K Dequantization ---

function dequantize_q5_k(data::AbstractVector{UInt8}, num_elements::Int)
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    # block_q5_K:
    #   d, dmin: 2*2 = 4 bytes
    #   scales: 12 bytes
    #   qh: QK_K/8 = 32 bytes
    #   qs: QK_K/2 = 128 bytes
    # Total = 4 + 12 + 32 + 128 = 176 bytes per block.
    
    d_ref = Ref{UInt8}(0)
    m_ref = Ref{UInt8}(0)
    
    for i in 1:nb
        base = (i-1) * 176
        d_val = Float32(reinterpret(Float16, (UInt16(data[base+2]) << 8) | UInt16(data[base+1])))
        m_val = Float32(reinterpret(Float16, (UInt16(data[base+4]) << 8) | UInt16(data[base+3])))
        
        scales = @view data[base+5:base+16]
        qh = @view data[base+17:base+48]
        qs = @view data[base+49:base+176]
        
        is_idx = 0
        ql_idx = 0
        qh_idx = 0
        for j in 0:64:255
            get_scale_min_k4!(is_idx, scales, d_ref, m_ref)
            d1 = d_val * d_ref[]
            m1 = m_val * m_ref[]
            
            get_scale_min_k4!(is_idx + 1, scales, d_ref, m_ref)
            d2 = d_val * d_ref[]
            m2 = m_val * m_ref[]
            
            # Load 8 bytes (64 bits) from qh for 64 elements
            hm1 = (UInt32(qh[qh_idx+1])) | (UInt32(qh[qh_idx+2]) << 8) | (UInt32(qh[qh_idx+3]) << 16) | (UInt32(qh[qh_idx+4]) << 24)
            hm2 = (UInt32(qh[qh_idx+5])) | (UInt32(qh[qh_idx+6]) << 8) | (UInt32(qh[qh_idx+7]) << 16) | (UInt32(qh[qh_idx+8]) << 24)
            
            for l in 0:31
                ql_val = qs[ql_idx + l + 1]
                
                # Interleaved bits? Llama.cpp: m = 1 << k
                # b0 = hm & m ? 16 : 0; m <<= 1
                # b1 = hm & m ? 16 : 0; m <<= 1
                
                b1 = (hm1 & (UInt32(1) << l)) != 0 ? 16.0f0 : 0.0f0
                b2 = (hm2 & (UInt32(1) << l)) != 0 ? 16.0f0 : 0.0f0
                
                y[(i-1)*256 + j + l + 1] = d1 * ((ql_val & 0xF) + b1) - m1
                y[(i-1)*256 + j + l + 33] = d2 * ((ql_val >> 4) + b2) - m2
            end
            ql_idx += 32
            qh_idx += 8
            is_idx += 2
        end
    end
    return y
end

# --- Q2_K Dequantization ---

function dequantize_q2_k(data::AbstractVector{UInt8}, num_elements::Int)
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    # block_q2_K:
    #   scales[16], qs[64], d, dmin
    # Total = 16 + 64 + 2 + 2 = 84 bytes per block.
    
    for i in 1:nb
        base = (i-1) * 84
        scales = @view data[base+1:base+16]
        qs = @view data[base+17:base+80]
        d = Float32(reinterpret(Float16, (UInt16(data[base+82]) << 8) | UInt16(data[base+81])))
        m = Float32(reinterpret(Float16, (UInt16(data[base+84]) << 8) | UInt16(data[base+83])))
        
        is_idx = 1
        q_off = 0
        for n in 0:128:255
            shift = 0
            for j in 0:3
                sc1 = scales[is_idx]
                dl1 = d * (sc1 & 0x0F); ml1 = m * (sc1 >> 4)
                for l in 0:15
                    y[(i-1)*256 + n + j*16 + l + 1] = dl1 * ((qs[q_off + l + 1] >> shift) & 3) - ml1
                end
                
                sc2 = scales[is_idx+1]
                dl2 = d * (sc2 & 0x0F); ml2 = m * (sc2 >> 4)
                for l in 0:15
                    y[(i-1)*256 + n + j*16 + l + 17] = dl2 * ((qs[q_off + l + 17] >> shift) & 3) - ml2
                end
                
                shift += 2
                is_idx += 2
            end
            q_off += 32
        end
    end
    return y
end

# --- Q3_K Dequantization ---

function dequantize_q3_k(data::AbstractVector{UInt8}, num_elements::Int)
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    # block_q3_K:
    #   hmask[32], qs[64], scales[12], d
    # Total = 32 + 64 + 12 + 2 = 110 bytes per block.
    
    for i in 1:nb
        base = (i-1) * 110
        hmask = @view data[base+1:base+32]
        qs = @view data[base+33:base+96]
        scales_raw = @view data[base+97:base+108]
        d_all = Float32(reinterpret(Float16, (UInt16(data[base+110]) << 8) | UInt16(data[base+109])))
        
        # Repack scales (12 bytes -> 16 scales)
        a0 = UInt32(scales_raw[1]) | (UInt32(scales_raw[2]) << 8) | (UInt32(scales_raw[3]) << 16) | (UInt32(scales_raw[4]) << 24)
        a1 = UInt32(scales_raw[5]) | (UInt32(scales_raw[6]) << 8) | (UInt32(scales_raw[7]) << 16) | (UInt32(scales_raw[8]) << 24)
        a2 = UInt32(scales_raw[9]) | (UInt32(scales_raw[10]) << 8) | (UInt32(scales_raw[11]) << 16) | (UInt32(scales_raw[12]) << 24)
        
        kmask1 = 0x03030303
        kmask2 = 0x0f0f0f0f
        
        s0 = (a0 & kmask2) | (((a2 >> 0) & kmask1) << 4)
        s1 = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4)
        s2 = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4)
        s3 = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4)
        
        repacked_scales = Vector{Int8}(undef, 16)
        for k_sc in 0:3
            repacked_scales[k_sc*4 + 1] = Int8(s0 >> (8*k_sc) & 0xFF)
            repacked_scales[k_sc*4 + 2] = Int8(s1 >> (8*k_sc) & 0xFF)
            repacked_scales[k_sc*4 + 3] = Int8(s2 >> (8*k_sc) & 0xFF)
            repacked_scales[k_sc*4 + 4] = Int8(s3 >> (8*k_sc) & 0xFF)
        end
        
        is_idx = 1
        q_off = 0
        m_bit = UInt8(1)
        for n in 0:128:255
            shift = 0
            for j in 0:3
                dl1 = d_all * (Float32(repacked_scales[is_idx]) - 32.0f0)
                for l in 0:15
                    y[(i-1)*256 + n + l + 1] = dl1 * (Float32((qs[q_off + l + 1] >> shift) & 3) - ((hmask[l + 1] & m_bit) != 0 ? 0.0f0 : 4.0f0))
                end
                is_idx += 1
                
                dl2 = d_all * (Float32(repacked_scales[is_idx]) - 32.0f0)
                for l in 0:15
                    y[(i-1)*256 + n + l + 17] = dl2 * (Float32((qs[q_off + l + 17] >> shift) & 3) - ((hmask[l + 17] & m_bit) != 0 ? 0.0f0 : 4.0f0))
                end
                is_idx += 1
                
                shift += 2
                m_bit <<= 1
            end
            q_off += 32
        end
    end
    return y
end


# --- Q6_K Dequantization ---

function dequantize_q6_k(data::AbstractVector{UInt8}, num_elements::Int)
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    # block_q6_K:
    #   ql[128], qh[64], scales[16], d
    # Total = 128 + 64 + 16 + 2 = 210 bytes per block.
    
    for i in 1:nb
        base = (i-1) * 210
        ql = @view data[base+1:base+128]
        qh = @view data[base+129:base+192]
        scales = @view data[base+193:base+208]
        d = Float32(reinterpret(Float16, (UInt16(data[base+210]) << 8) | UInt16(data[base+209])))
        
        # 2 super-blocks of 128
        ql_off = 0
        qh_off = 0
        sc_off = 0
        y_off = (i-1)*256
        for n in 0:128:255
            for l in 0:31
                is = l ÷ 16
                q1 = Int8((ql[ql_off + l + 1] & 0x0F) | ((qh[qh_off + l + 1] & 3) << 4)) - 32
                q2 = Int8((ql[ql_off + l + 33] & 0x0F) | (((qh[qh_off + l + 1] >> 2) & 3) << 4)) - 32
                q3 = Int8((ql[ql_off + l + 1] >> 4) | (((qh[qh_off + l + 1] >> 4) & 3) << 4)) - 32
                q4 = Int8((ql[ql_off + l + 33] >> 4) | (((qh[qh_off + l + 1] >> 6) & 3) << 4)) - 32
                
                y[y_off + n + l + 1] = d * Float32(Int8(scales[sc_off + is + 1])) * Float32(q1)
                y[y_off + n + l + 33] = d * Float32(Int8(scales[sc_off + is + 3])) * Float32(q2)
                y[y_off + n + l + 65] = d * Float32(Int8(scales[sc_off + is + 5])) * Float32(q3)
                y[y_off + n + l + 97] = d * Float32(Int8(scales[sc_off + is + 7])) * Float32(q4)
            end
            ql_off += 64
            qh_off += 32
            sc_off += 8
        end
    end
    return y
end

end # module
