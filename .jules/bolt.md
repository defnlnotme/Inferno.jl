## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-16 - [Inefficient GPU Array Initialization]
**Learning:** Using `oneArray(zeros(T, dims...))` in `oneAPI.jl` causes an allocation and zero-initialization on the CPU, followed by a synchronous PCIe host-to-device copy to the GPU. For frequent or large buffer initializations (e.g., in `MoE` or `KVCache`), this creates significant CPU memory pressure and PCIe bus overhead.
**Action:** Use `oneArray{T}(undef, dims...)` to allocate memory directly on the GPU, then use `fill!(arr, 0.0f0)` to initialize the memory via a GPU kernel, avoiding the CPU allocation and transfer entirely.
