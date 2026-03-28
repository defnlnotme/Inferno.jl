## 2025-05-16 - [GPU Array Initialization Pattern]
**Learning:** Initializing GPU arrays with `oneArray(zeros(T, dims...))` is a performance anti-pattern as it allocates and zeroes the memory on the CPU before a host-to-device transfer.
**Action:** Always use the optimized pattern `fill!(oneArray{T}(undef, dims...), zero(T))` to allocate directly on the GPU and initialize via a kernel, bypassing the CPU-to-GPU transfer bottleneck.
