using Inferno

file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

println("Looking for tokens in 'The capital of France is':")

# Find each word
for word in ["The", " The", "capital", " capital", "of", " of", "France", " France", "is", " is"]
    for (i, tok) in enumerate(tokens)
        if tok == word || tok == "Ġ" * word
            println("  '$word' -> token $(i-1): ", repr(tok))
            break
        end
    end
end

# Also try to find them with different prefixes
println("\nAll tokens containing 'The' (first 10):")
count = 0
for (i, tok) in enumerate(tokens)
    if occursin("The", tok) && length(tok) < 15
        println("  Token $(i-1): ", repr(tok))
        count += 1
        if count >= 10
            break
        end
    end
end

println("\nAll tokens containing 'capital' (first 10):")
count = 0
for (i, tok) in enumerate(tokens)
    if occursin("capital", lowercase(tok)) && length(tok) < 20
        println("  Token $(i-1): ", repr(tok))
        count += 1
        if count >= 10
            break
        end
    end
end
