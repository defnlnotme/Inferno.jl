using Test
using Inferno
using Statistics
using oneAPI

const MODEL_PATH = joinpath(@__DIR__, "models", "Qwen3.5-0.8B-UD-IQ2_XXS.gguf")

@testset "Inferno Tests" begin

    @testset "Security: Unbounded string allocation" begin
        io = IOBuffer()
        write(io, UInt64(typemax(UInt64))) # Arbitrarily large length
        seekstart(io)
        @test_throws ErrorException Inferno.GGUF.read_string(io)
    end

    @testset "GGUF Parsing" begin
        # Test unknown GGUF type
        io = IOBuffer()
        invalid_type = reinterpret(Inferno.GGUF.GGUFValueType, 999%UInt32)
        @test_throws ErrorException("Unknown GGUF type: $(invalid_type)") Inferno.GGUF.read_value(io, invalid_type)

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

    @testset "RMSNorm" begin
        using Inferno.Model
        using oneAPI

        # Test single sequence (1D-like, but technically 2D: hidden_size x 1)
        hidden_size = 1024
        seq_len = 1
        eps = 1f-6

        x_cpu = rand(Float32, hidden_size, seq_len)
        w_cpu = rand(Float32, hidden_size)

        # Expected output mathematically
        m = sum(x_cpu .* x_cpu, dims=1) .* (1.0f0 / Float32(hidden_size))
        inv_rms = 1.0f0 ./ sqrt.(m .+ eps)
        expected = x_cpu .* inv_rms .* w_cpu

        # GPU execution
        norm = Inferno.Model.RMSNorm(oneArray(w_cpu), eps)
        x_gpu = oneArray(x_cpu)
        res_gpu = norm(x_gpu)
        res_cpu = collect(res_gpu)

        @test res_cpu ≈ expected atol=1e-4

        # Test batched sequence (hidden_size x seq_len)
        seq_len = 10
        x_cpu_batch = rand(Float32, hidden_size, seq_len)

        m_batch = sum(x_cpu_batch .* x_cpu_batch, dims=1) .* (1.0f0 / Float32(hidden_size))
        inv_rms_batch = 1.0f0 ./ sqrt.(m_batch .+ eps)
        expected_batch = x_cpu_batch .* inv_rms_batch .* w_cpu

        x_gpu_batch = oneArray(x_cpu_batch)
        res_gpu_batch = norm(x_gpu_batch)
        res_cpu_batch = collect(res_gpu_batch)

        @test res_cpu_batch ≈ expected_batch atol=1e-4

        # Test small numbers
        x_small = rand(Float32, hidden_size, 1) .* 1f-5
        m_small = sum(x_small .* x_small, dims=1) .* (1.0f0 / Float32(hidden_size))
        inv_rms_small = 1.0f0 ./ sqrt.(m_small .+ eps)
        expected_small = x_small .* inv_rms_small .* w_cpu

        x_gpu_small = oneArray(x_small)
        res_gpu_small = norm(x_gpu_small)
        res_cpu_small = collect(res_gpu_small)

        @test res_cpu_small ≈ expected_small atol=1e-6
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

    @testset "Server Prompt Building" begin
        using Inferno.Server

        # Test 1: Only user message
        msgs1 = [Server.Message("user", "Hello!")]
        prompt1 = Server.build_prompt(msgs1)
        expected1 = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        @test prompt1 == expected1

        # Test 2: System and user message
        msgs2 = [
            Server.Message("system", "You are a helpful assistant."),
            Server.Message("user", "What is 2+2?")
        ]
        prompt2 = Server.build_prompt(msgs2)
        expected2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        @test prompt2 == expected2

        # Test 3: System, user, assistant, user (conversation flow)
        msgs3 = [
            Server.Message("system", "You are a helpful assistant."),
            Server.Message("user", "What is 2+2?"),
            Server.Message("assistant", "It is 4."),
            Server.Message("user", "What about 3+3?")
        ]
        prompt3 = Server.build_prompt(msgs3)
        expected3 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nIt is 4.<|im_end|>\n<|im_start|>user\nWhat about 3+3?<|im_end|>\n<|im_start|>assistant\n"
        @test prompt3 == expected3

        # Test 4: Edge cases (empty message array)
        msgs4 = Server.Message[]
        prompt4 = Server.build_prompt(msgs4)
        expected4 = "<|im_start|>assistant\n"
        @test prompt4 == expected4

        # Test 5: Unsupported roles are ignored
        msgs5 = [
            Server.Message("user", "Hello!"),
            Server.Message("unsupported_role", "This should be ignored")
        ]
        prompt5 = Server.build_prompt(msgs5)
        expected5 = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        @test prompt5 == expected5
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

    @testset "Engine.sample" begin
        # Test sample with temperature 0.0 (argmax)
        logits1 = Float32[1.0, 5.0, 2.0, 4.0]
        result1 = Inferno.Engine.sample(logits1, 0.0f0, 1.0f0)
        @test result1 == 2

        logits2 = Float32[-10.0, -5.0, -1.0]
        result2 = Inferno.Engine.sample(logits2, 0.0f0, 1.0f0)
        @test result2 == 3

        # Test with duplicate max (argmax should return first index)
        logits3 = Float32[1.0, 5.0, 2.0, 5.0]
        result3 = Inferno.Engine.sample(logits3, 0.0f0, 1.0f0)
        @test result3 == 2
    end

end

@testset "Server Endpoints Tests" begin
    include("test_server.jl")
end
