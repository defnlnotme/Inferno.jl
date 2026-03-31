using Inferno

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check the tokenizer's indexing
prompt = "The"
tokens = Inferno.Tokenizer.encode(tok, prompt)
println("Tokens from encode: ", tokens)

# Check what the tokenizer thinks about special tokens
println("\nSpecial tokens:")
println("  BOS ID: ", tok.bos_id)
println("  EOS ID: ", tok.eos_id)

# The tokenizer stores tokens as 0-indexed (from GGUF)
# But Julia arrays are 1-indexed
# So we need to add 1 when indexing into embedding matrix

# Let's verify by checking what "The" decodes to
println("\nDecode test:")
println("  Decode [761]: '", Inferno.Tokenizer.decode(tok, [761]), "'")
println("  Decode [762]: '", Inferno.Tokenizer.decode(tok, [762]), "'")

# Check if the forward function handles this correctly
# In forward_cpu!, we have: x = view(model.embed, :, tok)
# This is WRONG if tok is 0-indexed!
# We need: x = view(model.embed, :, tok + 1)
