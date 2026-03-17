using Test
using Inferno
using Inferno.GGUF
using Inferno.Loader
using Inferno.Dequant
using Statistics
using oneAPI

const MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "..", "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))

@testset "BISECT STAGE 1: GGUF & Weight Loading Audit" begin
    println("\n=== Auditing GGUF: $MODEL_PATH ===")
    flush(stdout)
    
    @test isfile(MODEL_PATH)
    file = read_gguf(MODEL_PATH)
    
    @testset "Tensor Existence & Metadata" begin
        # Check for core Qwen tensors
        cores = ["token_embd.weight", "blk.0.attn_qkv.weight", "output_norm.weight", "output.weight"]
        for c in cores
            @test haskey(file.tensors, c)
            info = file.tensors[c]
            println("  - $c: dims=$(info.dimensions), type=$(info.type)")
        end
    end

    @testset "Dequantization Integrity (Statistical Audit)" begin
        # Audit a few layers
        layers_to_check = [0, 5, 10]
        tensors_to_check = ["attn_qkv.weight", "ffn_up.weight", "ffn_down.weight"]
        
        for i in layers_to_check
            for t_base in tensors_to_check
                name = "blk.$i.$t_base"
                if !haskey(file.tensors, name)
                    println("  Skipping missing tensor: $name")
                    continue
                end
                
                info = file.tensors[name]
                println("  Auditing $name ($(info.type))...")
                
                # Extract using Loader (which performs dequantization)
                weights = Loader.extract_tensor(file, name)
                
                # 1. NaN/Inf Check
                nan_count = count(isnan, weights)
                inf_count = count(isinf, weights)
                @test nan_count == 0
                @test inf_count == 0
                
                # 2. Statistical Range Check
                # Use Float64 for stats to avoid precision issues with large matrices
                weights64 = Float64.(weights)
                w_min = minimum(weights64)
                w_max = maximum(weights64)
                w_mean = mean(weights64)
                w_std = std(weights64)
                
                println("    Range: [$(round(w_min, digits=4)), $(round(w_max, digits=4))]")
                println("    Mean: $(round(w_mean, digits=6)), Std: $(round(w_std, digits=6))")
                flush(stdout)
                
                @test abs(w_mean) < 1.0     # Mean should be near 0
                @test w_std > 1e-6          # Should not be all zeros
                @test abs(w_min) < 500.0    # No extreme outliers
                @test abs(w_max) < 500.0
                
                # 3. Flatness Check
                # If too many values are EXACTLY the same, dequant might be stuck
                zero_count = count(x -> x == 0.0f0, weights)
                zero_ratio = zero_count / length(weights)
                println("    Zero Ratio: $(round(zero_ratio * 100, digits=2))%")
                @test zero_ratio < 0.95     # Unless it's a very sparse model
            end
        end
    end

    @testset "GPU Upload Integrity" begin
        # For IQ2_XXS, we check the GPU struct directly
        # For others, we check the oneArray conversion
        
        name = "token_embd.weight"
        info = file.tensors[name]
        println("  Checking GPU upload for $name...")
        
        if info.type == GGUF.GGML_TYPE_IQ2_XXS
            # This follows the optimized path in Loader.jl:47
            w_gpu = Loader.get_weight(file, name)
            @test w_gpu isa Inferno.Model.IQ2XXSMatrix
            @test length(w_gpu.data) > 0
            println("    Verfied IQ2_XXS GPU Matrix")
        else
            # Dense path
            w_gpu = Loader.get_weight(file, name)
            @test w_gpu isa oneArray
            
            # Check if GPU data is finite
            w_cpu = collect(w_gpu)
            @test all(isfinite, w_cpu)
            println("    Verified Dense GPU weight (finite: OK)")
        end
    end
end
