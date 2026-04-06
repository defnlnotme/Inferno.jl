## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2026-03-23 - [Eliminating Synchronous CPU-GPU Roundtrips]
**Learning:** Manual scalar indexing and explicit `Array(gpu_array)` copies in the inference hot-path (e.g., for Softmax, SiLU, or Gating) force synchronous CPU-GPU roundtrips. This introduces massive latency because the CPU must wait for the GPU to finish, copy data over the PCIe bus, perform a slow serial loop, and copy back.
**Action:** Use GPU-native vectorized broadcasting (`@.`) and device-side reductions (`maximum`, `sum`) to keep execution entirely on the GPU. For buffer initialization, use the `fill!(oneArray{T}(undef, ...), zero(T))` pattern to allocate directly on-device.
