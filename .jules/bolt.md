## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2026-05-06 - [LoopVectorization Field Access Constraint]
**Learning:** `LoopVectorization.jl` (`@turbo`) does not support direct struct field access (e.g., `attn.head_dim`) within the loop body because it operates on the AST before property access is optimized. This causes compilation errors or prevents vectorization.
**Action:** Always hoist necessary struct fields to local variables before the `@turbo` block.
