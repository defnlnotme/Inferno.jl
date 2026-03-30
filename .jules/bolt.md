## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-15 - [Host-Device Synchronization Bottlenecks]
**Learning:** Using `Array(gpu_array)` in hot paths like `softmax_kernel!` and `MLP` forward passes triggered synchronous CPU-GPU transfers for every single token. This creates a massive PCIe bottleneck and stalls the GPU pipeline.
**Action:** Always use GPU-native vectorized broadcasting (`@.`) and device-side reductions (`maximum`, `sum`) to keep the entire inference loop on the GPU.
