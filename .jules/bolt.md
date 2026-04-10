## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-16 - [Elimination of CPU-GPU Hot-Path Synchronization]
**Learning:** `softmax_kernel!`, `MLP`, and `FullAttention` gating in `src/Model.jl` used `Array()` and `copyto!`, causing expensive synchronous round-trips to the host for every token. Using GPU-native broadcasting (`@.`) and reductions (`maximum`, `mapreduce`) with Float32 upcasting entirely eliminates these bottlenecks while maintaining numerical stability.
**Action:** Always implement activations and normalization kernels using Julia's vectorized broadcasting and device-side reductions to keep the inference hot-path on the GPU.
