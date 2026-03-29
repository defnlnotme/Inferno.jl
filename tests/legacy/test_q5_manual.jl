using Inferno

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

gate_q = model_q.layers[2].mlp.gate_weight

# Get the raw data for block 400
block_idx = 400
block_offset = block_idx * 176 + 1

# Parse the block structure
d = Float32(reinterpret(Float16, gate_q.data[block_offset:block_offset+1])[1])
dmin = Float32(reinterpret(Float16, gate_q.data[block_offset+2:block_offset+3])[1])
scales = gate_q.data[block_offset+4:block_offset+15]
qh = gate_q.data[block_offset+16:block_offset+47]
qs = gate_q.data[block_offset+48:block_offset+175]

println("Block 400 structure:")
println("  d = $d")
println("  dmin = $dmin")
println("  scales (12 bytes): ", scales)
println("  qh (32 bytes, first 10): ", qh[1:10])
println("  qs (128 bytes, first 10): ", qs[1:10])

# Manually compute values for il=10 (indices 160-175)
println("\n=== Manual calculation for il=10 ===")
il = 10
is_val = (il ÷ 4) * 2  # = 4
q_offset = 32 * (il ÷ 4) + 16 * (il % 2)  # = 32*2 + 16*0 = 64
qh_offset = 16 * (il % 2)  # = 0
ul = UInt8(1 << (il ÷ 2))  # = 1 << 5 = 32
il_mod = il % 4  # = 2
k = il_mod ÷ 2  # = 1

println("is = $is_val, q_offset = $q_offset, qh_offset = $qh_offset, ul = $ul, il_mod = $il_mod, k = $k")

# Scale extraction for is >= 4
println("\nScale extraction (j=4, k=1):")
println("  scales[10] = $(scales[10])")
println("  scales[2] = $(scales[2])")
println("  scales[6] = $(scales[6])")

# Scale formula: (scales[10] & 0x0f) | ((scales[2] & 0xc0) >> 2)
sc_manual = (UInt8(scales[10]) & 0x0f) | ((UInt8(scales[2]) & 0xc0) >> 2)
println("  sc (manual) = $sc_manual")

# Min formula: (scales[10] >> 4) | ((scales[6] & 0xc0) >> 2)
m_manual = (UInt8(scales[10]) >> 4) | ((UInt8(scales[6]) & 0xc0) >> 2)
println("  m (manual) = $m_manual")

# Now compute dl and ml
# d_eff = il_mod < 2 ? d : d / 16
d_eff = il_mod < 2 ? d : d / 16.0f0
println("\nd_eff = $d_eff")
dl = d_eff * Float32(sc_manual)
ml = dmin * Float32(m_manual)
println("dl = $dl, ml = $ml")

# Process 16 values for il=10
println("\nValues for il=10 (indices 160-175):")
mask = il_mod < 2 ? 0x0f : 0xf0
qh_add = il_mod < 2 ? 16.0f0 : 256.0f0
println("mask = $mask, qh_add = $qh_add")

for i in 0:15
 q_val = qs[q_offset + i + 1]
 qh_val = qh[qh_offset + i + 1]
 
 low = q_val & mask
 if mask == 0xf0
 low = low >> 4
 end
 
 high = (qh_val & ul) != 0 ? qh_add : 0.0f0
 v = Float32(low) + high
 result = dl * v - ml
 
 # Expected from float
 expected = model_f.layers[2].mlp.gate_weight[101, 160 + i + 1]
 diff = abs(result - expected)
 
 println("  i=$i: q=$q_val, qh=$qh_val, low=$low, high=$high, v=$v, result=$result, expected=$expected, diff=$diff")
end
