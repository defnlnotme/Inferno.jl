## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-16 - [CPU-GPU Round-trip Bottlenecks]
**Learning:** Core kernels like Softmax, MLP gating, and MoE activations were repeatedly using `Array()` or `collect()` to pull GPU data to the host for simple operations. This introduces massive PCIe latency and synchronizes the GPU, breaking the execution pipeline.
**Action:** Always prefer GPU-native broadcasting (`@.`) and reductions (`mapreduce`) on `oneArray` types. For `Float16` operations requiring precision or stability (like `exp`), upcast to `Float32` within the GPU broadcast to keep computation on-device.
