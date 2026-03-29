# Debug script to verify scale extraction
# In C (0-indexed):
# if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
# else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
#
# Converting to Julia (1-indexed):
# q[j] → scales[j + 1]
# q[j+4] → scales[j + 5]
# q[j-4] → scales[j - 3]  (since j-4 in 0-indexed = j-4+1 = j-3 in 1-indexed)
# q[j-0] = q[j] → scales[j + 1]

# So the correct translation for j >= 4:
# *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
# → (scales[j + 5] & 0x0f) | ((scales[j - 3] >> 6) << 4)
#
# *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)
# → (scales[j + 5] >> 4) | ((scales[j + 1] >> 6) << 4)

# Wait, q[j-0] = q[j], so in Julia it's scales[j + 1]
# NOT scales[j - 3 + 1] = scales[j - 2]

println("Testing scale extraction logic")

# Original code had:
# sc1 = (scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4)
# m1 = (scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4)
#
# But is_idx in Julia = j in C (since is_idx = 2*j where j is 0,1,2,3)
# So for is_idx >= 4 (which means is_idx can be 4,5,6):
# The correct formula should be:
# sc = (scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4)  ✓
# m = (scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4)  ✓
#
# Wait, that was already correct! Let me re-check...

# Actually wait - in the original Inferno code, is_idx = 2*j where j goes 0,1,2,3
# So is_idx goes 0, 2, 4, 6
# For j=2, is_idx=4, which is >= 4, so we use the else branch
# For j=3, is_idx=6, which is >= 4, so we use the else branch

# In the else branch for sc1:
# Original: UInt8((scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4))
# Correct:  UInt8((scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4))
# 
# Hmm, is_idx - 3 when is_idx=4 gives us 1, which corresponds to q[0] in C
# But we need q[is_idx-4] = q[0], which is q[0] = scales[1] in Julia
# scales[is_idx - 3] = scales[4 - 3] = scales[1] ✓

# For m1:
# Original: UInt8((scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4))
# Correct:  UInt8((scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4))
#
# is_idx + 1 when is_idx=4 gives us 5, which corresponds to q[4] in C
# But we need q[is_idx-0] = q[is_idx] = q[4], which is q[4] = scales[5] in Julia
# scales[is_idx + 1] = scales[4 + 1] = scales[5] ✓

# So the original code was actually correct! The issue must be elsewhere.

println("Scale extraction appears correct after analysis.")
