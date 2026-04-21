using Inferno
using Inferno.Gemma4
using Inferno.Gemma4Loader
using Inferno.Tokenizer: encode, decode

model, tok = Gemma4Loader.load_gemma4("test/models/gemma-4-E2B-it"; max_seq_len=512)

# Chat template for Gemma4
prompt = "<start_of_turn>user\nWhat is 2 + 2?<end_of_turn>\n<start_of_turn>model\n"
println("Prompt: ", repr(prompt))

# Encode
token_ids = encode(tok, prompt)
println("Encoded: ", length(token_ids), " tokens, first 10: ", token_ids[1:min(10, length(token_ids))])

# Generate
stop_tokens = Set{Int}([107])  # <end_of_turn> (0-indexed) = 108 (1-indexed)
generated = Gemma4.generate(model, token_ids, 50; temperature=0.7f0, top_k=50, stop_tokens=stop_tokens)

# Decode
output_text = decode(tok, generated)
println("Generated tokens: ", generated[1:min(20, length(generated))])
println("Generated text: ", output_text)
