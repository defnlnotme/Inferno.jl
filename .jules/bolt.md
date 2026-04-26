## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-15 - [O(N log k) Sampling and Zero-Alloc Multithreading]
**Learning:** For large vocabularies (e.g. Qwen's 151k+), a full `sortperm` during sampling is a significant bottleneck. Using `partialsortperm` reduces this to O(N log k). Additionally, multithreaded GEMV in the `lm_head` can be made zero-allocation by using `@sync`/`@spawn` with direct writes to disjoint `view`s of the output buffer, avoiding the need for intermediate task buffers and subsequent copies.
**Action:** Always prefer `partialsortperm` for top-k filtering and use direct-view writes for parallelized matrix-vector operations.
