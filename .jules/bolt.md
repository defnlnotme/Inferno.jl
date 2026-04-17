## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2026-04-17 - [Synchronous CPU-GPU Round-trips]
**Learning:** In the inference hot-path, calling `Array(gpu_array)` or `collect(gpu_array)` triggers a synchronous transfer from GPU VRAM to Host RAM, blocking the CPU until the transfer is complete. Subsequent operations on the CPU followed by `copyto!` or `oneArray()` further increase latency and memory bandwidth pressure.
**Action:** Use Julia's broadcasting (`@.`), device-side reductions (`maximum`, `mapreduce`), and BLAS wrappers (`mat_mul_AT_B!`) to keep data and computation entirely on the GPU. Always ensure `T = eltype(x)` is defined and views are correctly reshaped to 2D for GEMM operations.
