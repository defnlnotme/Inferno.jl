using Inferno

function main()
    file = GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    println("Token 1 (used in test): ", repr(tokens[2]))
    println("Token 562 (\" The\"): ", repr(tokens[563]))

    # Find "The" or " The"
    println("\nSearching for 'The' variants:")
    for (i, tok) in enumerate(tokens)
        if occursin("The", tok) && length(tok) < 10
            println("  Token $(i-1): ", repr(tok))
        end
    end

    println("\nFirst 30 tokens:")
    for i in 1:30
        println("  Token $(i-1): ", repr(tokens[i]))
    end
end

main()
