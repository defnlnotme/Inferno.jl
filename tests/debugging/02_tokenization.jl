using Test
using Inferno
using Inferno.GGUF
using Inferno.Tokenizer

const MODEL_PATH = get(ENV, "INFERNO_MODEL", "unsloth/Qwen3.5-0.8B-GGUF")

@testset "BISECT STAGE 2: Tokenization Audit" begin
    println("\n=== Auditing Tokenizer for: $MODEL_PATH ===")
    
    @test isfile(MODEL_PATH)
    file = read_gguf(MODEL_PATH)
    
    # 1. Load Tokenizer
    tokenizer = load_tokenizer(file.metadata)
    vocab_size = length(tokenizer.id_to_token)
    println("  Vocab Size: $vocab_size")
    @test vocab_size > 0
    
    # 2. Check Special Tokens
    println("  Special Tokens:")
    println("    BOS: $(tokenizer.bos_id)")
    println("    EOS: $(tokenizer.eos_id)")
    
    # We check if they are within vocab range
    @test tokenizer.bos_id === nothing || tokenizer.bos_id < vocab_size
    @test tokenizer.eos_id === nothing || tokenizer.eos_id < vocab_size
    
    # 3. Encoding/Decoding Round-Trip
    test_texts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "1 + 1 = 2",
        "Julia is a high-level, high-performance, dynamic programming language.",
        "你好，世界！" # Multi-byte characters
    ]
    
    for text in test_texts
        print("  Testing round-trip: \"$text\"... ")
        tokens = encode(tokenizer, text)
        @test !isempty(tokens)
        
        decoded = decode(tokenizer, tokens)
        # Some tokenizers append spaces or handling leading/trailing whitespace differently
        # but for Qwen it should be fairly close.
        @test strip(decoded) == strip(text)
        println("OK ($(length(tokens)) tokens)")
    end
    
    # 4. Out-of-Vocabulary / Edge Cases
    println("  Testing edge cases...")
    empty_tokens = encode(tokenizer, "")
    @test isempty(empty_tokens)
    
    # Long text
    long_text = repeat("token ", 10)
    tokens = encode(tokenizer, long_text)
    @test length(tokens) >= 10
end
