## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-16 - [LM Head Allocation Bottleneck]
**Learning:** The `lm_head_project!` function in `src/ModelCPU.jl` was allocating intermediate chunk buffers (~600KB per token) and performing redundant copy operations. This created significant GC pressure and overhead during the largest matrix-vector multiplication in the inference loop.
**Action:** Use `@sync` and `Threads.@spawn` to compute results directly into views of the pre-allocated output vector, eliminating intermediate allocations.
