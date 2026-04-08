## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-16 - [GPU Allocation Anti-Pattern]
**Learning:** Using `oneArray(zeros(T, dims...))` in Julia/oneAPI.jl is an anti-pattern. It allocates a zeroed array on the CPU first, then performs a synchronous host-to-device transfer. This wastes CPU cycles and PCIe bandwidth.
**Action:** Always use `fill!(oneArray{T}(undef, dims...), zero(T))` to allocate uninitialized memory directly on the GPU and zero it via a GPU kernel.
