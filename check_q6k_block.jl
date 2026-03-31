using Inferno
using Inferno.GGUF

file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

emb_info = file.tensors["token_embd.weight"]
offset = Int(emb_info.offset)
block_size = 210

# Read first block
one_block = @view file.tensor_data[offset:offset+block_size-1]

# Parse block_q6_K structure
# ql: 128 bytes (offset 0-127)
# qh: 64 bytes (offset 128-191)
# sc: 16 bytes (offset 192-207)
# d: 2 bytes (offset 208-209)

ql = @view one_block[1:128]
qh = @view one_block[129:192]
sc_bytes = one_block[193:208]
d_bytes = one_block[209:210]

println("=== First Q6_K Block ===")
println("Scale d bytes: ", d_bytes, " -> ", reinterpret(Float16, d_bytes))
println("Scales bytes: ", sc_bytes)
println("Scales as Int8: ", reinterpret(Int8, sc_bytes))

# Check if scale d is actually at the right position
# Maybe the structure is different?
println("\n=== Checking different interpretations ===")

# What if d is at the beginning?
d_alt = reinterpret(Float16, one_block[1:2])
println("d at beginning: ", d_alt)

# Check if block structure might be different
println("\n=== All zero check ===")
println("ql all zeros: ", all(==(0x00), ql))
println("qh all zeros: ", all(==(0x00), qh))
