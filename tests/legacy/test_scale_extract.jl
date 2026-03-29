# Test scale extraction
scales = UInt8[0xa6, 0xea, 0xb3, 0xb0, 0xa8, 0xe6, 0xb8, 0xb9, 0x96, 0xee, 0x32, 0xe7]

# For j=4, k=1 (in 0-indexed)
# Scale: (scales[9] & 0xF) | ((scales[1] & 0xc0) >> 2)
# Min: (scales[9] >> 4) | ((scales[5] & 0xc0) >> 2)
# In 1-indexed: scales[10], scales[2], scales[6]

println("Scale extraction test:")
println("scales[10] = $(scales[10]) (0x$(string(scales[10], base=16)))")
println("scales[2] = $(scales[2]) (0x$(string(scales[2], base=16)))")
println("scales[6] = $(scales[6]) (0x$(string(scales[6], base=16)))")

# C code: (q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)
# j=4, k=1: (q[9] & 0xF) | ((q[1] & 0xc0) >> 2)
# In Julia: (scales[10] & 0x0f) | ((scales[2] & 0xc0) >> 2)

sc = (scales[10] & 0x0f) | ((scales[2] & 0xc0) >> 2)
println("\nCalculated sc = $sc")

# Let's verify step by step
println("\nStep by step:")
println("  scales[10] & 0x0f = $(scales[10] & 0x0f)")
println("  scales[2] & 0xc0 = $(scales[2] & 0xc0)")
println("  (scales[2] & 0xc0) >> 2 = $((scales[2] & 0xc0) >> 2)")
println("  sc = $((scales[10] & 0x0f) | ((scales[2] & 0xc0) >> 2))")

# But wait, let me re-read the Metal code...
# The Metal code uses 'q' which is 'xb->scales', passed to get_scale_min_k4_just2
# So the scales array is indexed directly

# Actually, I think I misread. Let me check what the scales array actually contains
println("\nFull scales array interpretation:")
for (i, s) in enumerate(scales)
 println("  scales[$i] = $s (0x$(string(s, base=16, pad=2)))")
end

# Check if the scale values make sense
# The scales should be small positive values (0-63 for the low 6 bits)
# or packed values for j >= 4

# Let me look at the expected values from the float model
# For row 100, the float values should give us a hint about the correct scale
