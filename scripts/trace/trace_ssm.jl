using Inferno
using Statistics
using LinearAlgebra

function main()
    cpu_model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    gpu_model, _ = load_model("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Get embedding for token 760 ("The")
    x_cpu = copy(view(cpu_model.embed, :, 760))
    x_gpu = copy(view(Float32.(gpu_model.embed), :, 760))

    println("Embedding check:")
    println("  CPU embed mean: ", mean(x_cpu), " std: ", std(x_cpu))
    println("  GPU embed mean: ", mean(x_gpu), " std: ", std(x_gpu))
    println("  Max diff: ", maximum(abs.(x_cpu - x_gpu)))

    # Check layer 1 in_norm
    cpu_in_norm = cpu_model.layers[1].in_norm
    gpu_in_norm = gpu_model.layers[1].in_norm

    h_cpu = cpu_in_norm(x_cpu)
    h_gpu = Float32.(gpu_in_norm(Float16.(x_cpu)))

    println("\nAfter in_norm:")
    println("  CPU mean: ", mean(h_cpu), " std: ", std(h_cpu))
    println("  GPU mean: ", mean(h_gpu), " std: ", std(h_gpu))
    println("  Max diff: ", maximum(abs.(h_cpu - h_gpu)))

    # Check SSM projections
    cpu_ssm = cpu_model.layers[1].op
    gpu_ssm = gpu_model.layers[1].op

    cpu_qkv = cpu_ssm.in_proj * h_cpu
    gpu_qkv = Float32.(gpu_ssm.in_proj * Float16.(h_cpu))

    println("\nQKV projection:")
    println("  CPU mean: ", mean(cpu_qkv), " std: ", std(cpu_qkv))
    println("  GPU mean: ", mean(gpu_qkv), " std: ", std(gpu_qkv))
    println("  Max diff: ", maximum(abs.(cpu_qkv - gpu_qkv)))

    cpu_z = cpu_ssm.gate_proj * h_cpu
    gpu_z = Float32.(gpu_ssm.gate_proj * Float16.(h_cpu))

    println("\nGate projection:")
    println("  CPU z mean: ", mean(cpu_z), " std: ", std(cpu_z))
    println("  GPU z mean: ", mean(gpu_z), " std: ", std(gpu_z))
    println("  Max diff: ", maximum(abs.(cpu_z - gpu_z)))
end

main()
