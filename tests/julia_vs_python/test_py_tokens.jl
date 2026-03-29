#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Python reference top-5: [220, 318, 26076, 95759, 271]
println("Python reference top tokens:")
for id in [220, 318, 26076, 95759, 271]
    println("  Token $id: '$(tokens[id + 1])'")
end
