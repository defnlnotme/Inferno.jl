## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2024-05-16 - [Fused RMSNorm and RoPE Optimization]
**Learning:** Fusing RMSNorm application and RoPE rotation into a single pass reduces memory bandwidth pressure significantly. Combined with `@turbo` for SIMD vectorization, this provided a measurable speedup in the attention hot path.
**Action:** Always look for opportunities to fuse point-wise operations and normalization with subsequent transformations to minimize memory round-trips.
