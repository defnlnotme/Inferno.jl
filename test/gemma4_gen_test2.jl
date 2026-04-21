using Inferno
using Inferno.Gemma4
using Inferno.Gemma4Loader

model, tok = Gemma4Loader.load_gemma4("test/models/gemma-4-E2B-it"; max_seq_len=512)

# Use correct Gemma4 token IDs (0-indexed, will be +1 in our system)
# <bos>=2, <|turn>=105, <turn|>=106, \n=107
# Template: <bos><|turn>user\n{content}<turn|>\n<|turn>model\n
# In our 1-indexed system: <bos>=3, <|turn>=106, <turn|>=107, \n=108

token_ids = [3, 106, 2365, 108, 3690, 564, 236744, 236779, 901, 236744, 236779, 236882, 107, 108, 106, 4369, 108]
println("Token IDs: ", token_ids, " (", length(token_ids), " tokens)")

# Generate with correct stop token (<turn|> = 107 in 0-indexed = 108 in 1-indexed)
stop_tokens = Set{Int}([107])  # <turn|> in 0-indexed
generated = Gemma4.generate(model, token_ids, 80; temperature=0.7f0, top_k=50, stop_tokens=stop_tokens)

println("Generated ", length(generated), " tokens")
println("Generated IDs (first 30): ", generated[1:min(30, length(generated))])

# Decode using HF tokenizer reference
println("\nNote: decode needs proper tokenizer - checking raw IDs")
# Check if any generated token is the stop token
for (i, t) in enumerate(generated)
    if t == 107
        println("Stop token found at position $i")
        break
    end
end
