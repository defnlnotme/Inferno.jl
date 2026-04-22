## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-16 - [Host-Device Synchronization Bottleneck]
**Learning:** Core GPU kernels like `softmax_kernel!` and `batched_softmax_kernel!` were implemented by copying data to the host, iterating over scalars on the CPU, and copying results back to the GPU. This "round-trip" pattern introduces massive latency and stalls the GPU pipeline for every token generation step.
**Action:** Use vectorized broadcasting (`@.`) and device-side reductions (`mapreduce` with `dims` specified) to keep all computations entirely on the GPU. Always ensure reductions return an array (by specifying `dims`) to avoid implicit host-device synchronization.
