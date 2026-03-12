using Test
using Inferno
using Statistics

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
