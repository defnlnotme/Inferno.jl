## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-16 - [Inefficient GPU Array Initialization]
**Learning:** Using `oneArray(zeros(T, dims...))` is a significant performance anti-pattern in `oneAPI.jl`. It forces a zero-filled allocation on the CPU followed by a full host-to-device transfer over the PCIe bus. For large model buffers and hot-path tensors, this introduces unnecessary latency and memory pressure.
**Action:** Always use `oneArray{T}(undef, dims...)` followed by `fill!(arr, 0.0f0)` to allocate directly on the GPU and initialize via a GPU kernel, completely bypassing the host.
