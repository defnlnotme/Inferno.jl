## 2025-05-15 - [Quantized Weight Memory Churn]
**Learning:** In `Inferno.jl`, `IQ2_XXS` weights were kept as `Vector{UInt8}` on the host, causing a full CPU-to-GPU copy of all model weights (multiple megabytes per layer) for every single token generated. This creates massive memory bandwidth pressure and significantly slows down inference.
**Action:** Move these weights to GPU VRAM once during loading in `Loader.jl` and update `Model.jl` to use `oneVector{UInt8}` directly in the `IQ2XXSMatrix` struct.

## 2025-05-15 - [GPU Kernel Synchronization Bottlenecks]
**Learning:** In `src/Model.jl`, several kernels (Softmax, RoPE, SiLU) and activations (MLP, MoE) were performing host-device round-trips using `Array()` and `copyto!()` or `collect()`. This stalls the GPU pipeline and saturates PCIe bandwidth. Using `mapreduce(..., dims=1)` ensures reductions stay on the device, and vectorized broadcasting (`@.`) over views allows for entirely on-device computation.
**Action:** Always prioritize keeping intermediate tensors on the GPU. Avoid scalar indexing and host-side loops. Use `Float32` upcasting within broadcasts for numerical stability when working with `Float16` arrays.
