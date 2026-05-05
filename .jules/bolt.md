## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2026-05-05 - [Fused RMSNorm and RoPE Vectorization]
**Learning:** Fusing RMSNorm scaling and RoPE application in `rmsnorm_rotary!` reduces memory bandwidth pressure. Applying `@turbo` from `LoopVectorization.jl` to these fused loops and to the QKV splitting loops in `FullAttentionCPU` provides significant CPU speedups by enabling SIMD vectorization.
**Action:** Always look for opportunities to fuse element-wise operations before memory-intensive passes, and utilize `@turbo` for nested loops in the inference hot-path.
