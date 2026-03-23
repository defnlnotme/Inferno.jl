## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-16 - [Inefficient GPU Array Initialization]
**Learning:** Using `oneArray(zeros(T, dims...))` creates a full copy of the array on the host (CPU) before transferring it to the device (GPU). For large model buffers (KV cache, MLP states), this creates unnecessary host memory pressure and adds significant latency to model initialization and inference hot-paths.
**Action:** Always use `oneArray{T}(undef, dims...)` followed by `fill!(arr, 0.0f0)` to allocate and initialize directly on the GPU using hardware-accelerated kernels.
