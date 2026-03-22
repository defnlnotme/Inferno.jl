using oneAPI
using Inferno
using Inferno.Model

function debug()
    T = Float16
    hidden_size = 1024
    x_cpu = randn(T, hidden_size) .* T(0.1)
    weight_cpu = ones(T, hidden_size)
    eps = T(1e-6)

    norm_func = Model.RMSNorm(weight_cpu, eps)

    println("--- CPU ---")
    y_cpu = norm_func(x_cpu)
    println("y_cpu has NaN: ", any(isnan, y_cpu))
    println("y_cpu mean: ", sum(y_cpu)/length(y_cpu))

    println("--- GPU ---")
    x_gpu = oneArray(x_cpu)
    weight_gpu = oneArray(weight_cpu)
    norm_func_gpu = Model.RMSNorm(weight_gpu, eps)

    y_gpu = norm_func_gpu(x_gpu)
    y_gpu_host = collect(y_gpu)
    println("y_gpu has NaN: ", any(isnan, y_gpu_host))
    println("y_gpu mean: ", sum(y_gpu_host)/length(y_gpu_host))

    # Try with larger values to trigger overflow
    println("--- GPU Overflow Test ---")
    x_cpu_large = randn(T, hidden_size) .* T(10.0)
    x_gpu_large = oneArray(x_cpu_large)
    y_gpu_large = norm_func_gpu(x_gpu_large)
    y_gpu_large_host = collect(y_gpu_large)
    println("y_gpu_large has NaN: ", any(isnan, y_gpu_large_host))
    println("First 5 of y_gpu_large: ", y_gpu_large_host[1:5])

    m_gpu = sum(abs2, x_gpu_large, dims=1) ./ T(size(x_gpu_large, 1))
    m_host = collect(m_gpu)
    println("m_gpu value: ", m_host[1])
end

debug()
