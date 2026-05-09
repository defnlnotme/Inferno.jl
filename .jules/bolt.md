## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2026-05-09 - [Large Vocabulary Sampling and Projection Bottlenecks]
**Learning:** For large vocabularies (e.g., Qwen 150k+), standard `sortperm` and `Set`-based filtering in `softmax_sample` create massive memory churn (~3MB/token). Similarly, `lm_head_project!` with intermediate buffers and redundant copies adds ~600KB/token overhead.
**Action:** Use `partialsortperm` for top-k selection and refactor parallel projections to write directly into output views using `@sync` and `Threads.@spawn`.
