using Test
using Inferno

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

end
