using Test
using Inferno
using Inferno.Model
using oneAPI
using Statistics

const MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "..", "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))

@testset "BISECT STAGE 3: Embedding & Norms Audit" begin
    println("\n=== Auditing Embedding & Norms for: $MODEL_PATH ===")
    
    # 1. Load Model
    model = load_model(MODEL_PATH)
    config = model.config
    println("  Embedding Hidden Size: $(config.hidden_size)")
    
    @testset "Embedding Access" begin
        # Check a few token IDs
        token_ids = [1, 100, 1000]
        for tid in token_ids
            emb = model.embed[:, tid]
            @test length(emb) == config.hidden_size
            @test all(isfinite, emb)
            @test std(emb) > 0.0 # Should not be all zeros
        end
        println("  Verified Embedding lookups (CPU).")
    end
    
    @testset "RMSNorm GPU Kernel Audit" begin
        # Test input: (hidden_size, batch=2)
        rows = config.hidden_size
        cols = 2
        x_cpu = randn(Float32, rows, cols)
        x_gpu = oneArray(x_cpu)
        
        # Norm weights
        norm = model.final_norm
        @test norm.weight isa oneArray
        
        # GPU forward
        y_gpu = norm(x_gpu)
        y_cpu = collect(y_gpu)
        
        # CPU verification
        # RMS = sqrt(mean(x^2))
        for j in 1:cols
            col = x_cpu[:, j]
            rms = sqrt(mean(col.^2) + norm.eps)
            expected = (col ./ rms) .* collect(norm.weight)
            
            # Comparison
            @test isapprox(y_cpu[:, j], expected, rtol=1e-5)
        end
        println("  Verified RMSNorm GPU kernel (Matches CPU reference).")
    end
    
    @testset "Embedding to GPU Flow" begin
        # In actual inference, tokens are converted to embeddings and moved to GPU
        # We simulate this.
        tokens = [1, 2, 3]
        # Look up on CPU
        emb_cpu = model.embed[:, tokens] # (hidden, seq)
        # Move to GPU
        emb_gpu = oneArray(emb_cpu)
        @test size(emb_gpu) == (config.hidden_size, 3)
        @test all(isfinite, collect(emb_gpu))
        println("  Verified Embedding to GPU flow.")
    end
end
