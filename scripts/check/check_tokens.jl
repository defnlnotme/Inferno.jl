using Inferno

function main()
    file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Check the tokens we're using
    prompt = "The capital of France is"

    # Find each token
    println("Looking for tokens in: \"$prompt\"")
    println()

    # Find "The" variations
    for (i, tok) in enumerate(tokens)
        if tok == "The" || tok == "ĠThe" || tok == "TheĠ"
            println("Found '$tok' at token $(i-1)")
        end
    end

    # Find "capital" variations
    for (i, tok) in enumerate(tokens)
        if occursin("capital", lowercase(tok)) && length(tok) < 15
            println("Found '$tok' at token $(i-1)")
        end
    end

    # Find "Paris" variations
    println()
    for (i, tok) in enumerate(tokens)
        if occursin("Paris", tok) || occursin("paris", tok)
            println("Found '$tok' at token $(i-1)")
        end
    end

    # Also check if there's a BOS token
    println("\nFirst 10 tokens:")
    for i in 1:10
        println("  Token $(i-1): ", repr(tokens[i]))
    end

    # Check special tokens
    println("\nSpecial tokens:")
    for key in keys(file.metadata)
        if occursin("special", lowercase(key)) || occursin("bos", lowercase(key)) || occursin("eos", lowercase(key))
            println("  $key: ", file.metadata[key])
        end
    end
end

main()
