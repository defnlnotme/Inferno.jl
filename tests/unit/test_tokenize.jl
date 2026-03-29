#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

# Load GGUF
file = read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Create a simple tokenizer
tok = SimpleTokenizer(file)

# Test tokenization - SimpleTokenizer uses 0-indexed IDs
text = "The"
ids = Inferno.Generate.encode(tok, text)
println("Tokenizing '$text':")
println("  Token IDs: ", ids)
for id in ids
    println("  Token $id: '$(tokens[id + 1])'")
end

# Compare with manual lookup
println("\nManual token lookup for 'The':")
for (i, t) in enumerate(tokens)
    if t == "The" || t == "ĠThe" || occursin("The", t)
        println("  Found at index $i: '$(t)'")
        if i > 600
            break
        end
    end
end
