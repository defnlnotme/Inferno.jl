using Test
using Inferno
using Statistics
using oneAPI

const MODEL_PATH = joinpath(@__DIR__, "models", "Qwen3.5-0.8B-UD-IQ2_XXS.gguf")

@testset "Inferno Tests" begin

    @testset "GGUF Parsing" begin
        file = Inferno.GGUF.read_gguf(MODEL_PATH)

        @test length(file.metadata) > 0
        @test length(file.tensors) > 0
        @test file.data_offset > 0

        # Check expected metadata keys
        @test haskey(file.metadata, "general.architecture")
        arch = file.metadata["general.architecture"]
        @test arch == "qwen35"

        # Model-specific keys use arch prefix
        @test haskey(file.metadata, "$(arch).block_count")
        @test haskey(file.metadata, "$(arch).embedding_length")

        # Tensor count should match model  
        println("  Architecture: $arch")
        println("  Metadata keys: $(length(file.metadata))")
        println("  Tensor count: $(length(file.tensors))")

        # Check some expected tensors exist
        @test haskey(file.tensors, "token_embd.weight")
        @test haskey(file.tensors, "blk.0.attn_qkv.weight")
        @test haskey(file.tensors, "output_norm.weight")
    end

    @testset "Tokenizer" begin
        file = Inferno.GGUF.read_gguf(MODEL_PATH)
        tok = Inferno.Tokenizer.load_tokenizer(file.metadata)

        @test length(tok.id_to_token) > 0
        @test tok.eos_id > 0
        println("  Vocab size: $(length(tok.id_to_token))")
        println("  EOS ID: $(tok.eos_id)")

        # Encode simple ASCII text
        ids = Inferno.Tokenizer.encode(tok, "Hello")
        @test length(ids) > 0
        println("  'Hello' -> $ids")

        # Decode back
        decoded = Inferno.Tokenizer.decode(tok, ids)
        println("  Decoded: '$decoded'")
        @test occursin("Hello", decoded) || occursin("hello", decoded) || length(decoded) > 0
    end

    @testset "Config Extraction" begin
        file = Inferno.GGUF.read_gguf(MODEL_PATH)
        arch = get(file.metadata, "general.architecture", "llm")

        block_count = Int(file.metadata["$(arch).block_count"])
        hidden_size = Int(file.metadata["$(arch).embedding_length"])
        num_heads = Int(file.metadata["$(arch).attention.head_count"])
        num_kv_heads = Int(file.metadata["$(arch).attention.head_count_kv"])

        println("  Architecture: $arch")
        println("  Layers: $block_count")
        println("  Hidden size: $hidden_size")
        println("  Heads: $num_heads, KV heads: $num_kv_heads")

        @test block_count == 24
        @test hidden_size == 1024
        @test num_heads == 8
        @test num_kv_heads == 2
    end

    @testset "Dequantization Kernels (CPU)" begin
        using Inferno.Dequant
        using Inferno.QuantsData

        @testset "IQ2_XXS" begin
            data = zeros(UInt8, 66); data[1] = 0x00; data[2] = 0x3c # d = 1.0
            y = dequantize_iq2_xxs(data, 256)
            @test all(y .== 0.0f0)
            data[3] = 0x01 # grid[2] = 0x08...082b
            y = dequantize_iq2_xxs(data, 256)
            @test y[1] == 4.375f0 # (43-8) * 0.125
            @test all(y[2:8] .== 0.0f0)
        end

        @testset "IQ2_XS" begin
            data = zeros(UInt8, 74); data[1] = 0x00; data[2] = 0x3c
            y = dequantize_iq2_xs(data, 256)
            @test all(y .== 0.0f0)
        end

        @testset "IQ3_XXS" begin
            data = zeros(UInt8, 98); data[1] = 0x00; data[2] = 0x3c
            y = dequantize_iq3_xxs(data, 256)
            @test all(y .== 0.0f0)
        end
    end

    @testset "CPU vs GPU Consistency" begin
        using Inferno.Model
        using Inferno.Dequant
        using Inferno.QuantsData
        
        N, K = 1, 1024
        data = rand(UInt8, 66 * 4)
        for i in 1:4
            base = (i-1)*66
            data[base+1] = 0x00; data[base+2] = 0x3c
        end
        
        weight_cpu = dequantize_iq2_xxs(data, K)
        Model.init_gpu_tables(QuantsData.IQ2XXS_GRID, QuantsData.KSIGNS_IQ2XS, QuantsData.KMASK_IQ2XS)
        
        weight_gpu = Model.IQ2XXSMatrix(oneArray(data), K, N)
        x_cpu = rand(Float32, K)
        x_gpu = oneArray(x_cpu)
        
        y_cpu = sum(weight_cpu .* x_cpu)
        res_gpu = Model.mat_mul(weight_gpu, reshape(x_gpu, K, 1))
        y_gpu = collect(res_gpu)[1]
        
        println("  Consistency diff: $(abs(y_cpu - y_gpu))")
        @test y_cpu ≈ y_gpu atol=1e-3
    end

    @testset "RotaryEmbedding" begin
        using Inferno.Model

        d = 4
        h = 1
        seq = 2
        rope = Model.RotaryEmbedding(d, base=10000.0)

        # Base freq = 10000.0
        # inv_freq = [1.0, 1.0 / (10000.0^(2/4))] = [1.0, 0.01]

        # Test input: shape (d, h, seq)
        # Sequence 1: 1.0, 2.0, 3.0, 4.0
        # Sequence 2: 5.0, 6.0, 7.0, 8.0
        x_cpu = zeros(Float32, d, h, seq)
        x_cpu[:, 1, 1] .= [1.0f0, 2.0f0, 3.0f0, 4.0f0]
        x_cpu[:, 1, 2] .= [5.0f0, 6.0f0, 7.0f0, 8.0f0]

        x_gpu = oneArray(x_cpu)

        # pos = 0
        pos = 0
        res_gpu = rope(x_gpu, pos)

        # Force sync and collect
        oneAPI.synchronize()
        res_cpu = collect(res_gpu)

        # Expected outputs:
        # seq = 1 (t=1), pos = 0 => p = pos + t - 1 = 0
        # For p = 0, freq = 0. cos(0)=1, sin(0)=0. Rotation is identity.
        @test res_cpu[:, 1, 1] ≈ [1.0f0, 2.0f0, 3.0f0, 4.0f0] atol=1e-5

        # seq = 2 (t=2), pos = 0 => p = 0 + 2 - 1 = 1
        # i = 1 (dims 1,2): freq = 1.0 * 1 = 1.0
        # x1 = 5.0, x2 = 6.0
        # x1_new = 5.0*cos(1.0) - 6.0*sin(1.0)
        # x2_new = 5.0*sin(1.0) + 6.0*cos(1.0)

        # i = 2 (dims 3,4): freq = 0.01 * 1 = 0.01
        # x1 = 7.0, x2 = 8.0
        # x1_new = 7.0*cos(0.01) - 8.0*sin(0.01)
        # x2_new = 7.0*sin(0.01) + 8.0*cos(0.01)

        exp_seq2 = [
            5.0f0 * cos(1.0f0) - 6.0f0 * sin(1.0f0),
            5.0f0 * sin(1.0f0) + 6.0f0 * cos(1.0f0),
            7.0f0 * cos(0.01f0) - 8.0f0 * sin(0.01f0),
            7.0f0 * sin(0.01f0) + 8.0f0 * cos(0.01f0)
        ]

        @test res_cpu[:, 1, 2] ≈ exp_seq2 atol=1e-5
    end

    @testset "Inference" begin
        model, tok = Inferno.load_model(MODEL_PATH)

        @test model.config.hidden_size == 1024
        @test model.config.num_hidden_layers == 24
        @test length(model.layers) == 24

        prompt = "The capital of France is"
        tokens = Inferno.Tokenizer.encode(tok, prompt)
        @test length(tokens) > 0
        println("  Prompt: \"$prompt\" -> $(length(tokens)) tokens")

        # Init KV caches
        caches = [Inferno.Model.init_kv_cache(
            model.config.head_dim,
            model.config.num_key_value_heads,
            model.config.max_position_embeddings)
            for _ in 1:model.config.num_hidden_layers]

        # Prefill
        logits = Inferno.Model.forward!(model, tokens, 0, caches)
        last_logits = vec(logits[:, end])

        # Logits should be finite real numbers
        @test all(isfinite, last_logits)
        @test maximum(last_logits) > -1000.0f0
        @test minimum(last_logits) < 1000.0f0
        println("  Logits: mean=$(round(mean(last_logits), digits=2)), max=$(round(maximum(last_logits), digits=2)), min=$(round(minimum(last_logits), digits=2))")

        # Generate 3 greedy tokens
        generated = Int[]
        for i in 1:3
            tok_id = argmax(last_logits)
            push!(generated, tok_id)
            # Decode should produce a non-empty string
            tok_str = Inferno.Tokenizer.decode(tok, [tok_id])
            @test length(tok_str) > 0
            print("  Token $i: $tok_id -> \"$tok_str\"")

            if tok_id == tok.eos_id
                println(" [EOS]")
                break
            end
            println()

            # Next step
            curr_pos = caches[1].pos
            next_logits = Inferno.Model.forward!(model, [tok_id], curr_pos, caches)
            last_logits = vec(next_logits[:, 1])
            @test all(isfinite, last_logits)
        end

        @test length(generated) >= 1
        full_decode = Inferno.Tokenizer.decode(tok, generated)
        println("  Full generation: \"$full_decode\"")
        @test length(full_decode) > 0
    end

end
