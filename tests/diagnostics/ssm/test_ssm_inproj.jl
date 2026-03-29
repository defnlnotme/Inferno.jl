using Inferno
using Statistics
using LinearAlgebra

function main()
    cpu_model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    gpu_model, _ = load_model("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Test with a specific input
    x = randn(Float32, 1024)
    x_cpu = copy(x)

    # CPU SSM layer
    cpu_ssm = cpu_model.layers[1].op

    # GPU SSM layer
    gpu_ssm = gpu_model.layers[1].op

    # Check in_proj weights
    println("CPU in_proj size: ", size(cpu_ssm.in_proj))
    println("GPU in_proj size: ", size(gpu_ssm.in_proj))

    # Test in_proj multiplication
    cpu_qkv = cpu_ssm.in_proj * x_cpu
    println("\nCPU in_proj * x: mean=$(mean(cpu_qkv)), std=$(std(cpu_qkv))")

    # Check if there's a transpose difference
    # GPU in_proj might be stored differently
    println("\nCPU in_proj[1:5, 1:5]:")
    println(cpu_ssm.in_proj[1:5, 1:5])

    println("\nGPU in_proj[1:5, 1:5]:")
    println(Float32.(gpu_ssm.in_proj[1:5, 1:5]))
end

main()
