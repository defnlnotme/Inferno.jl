using Inferno

_, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check token indexing
println("=== Token indexing check ===")

# Token 272 in 0-indexed = token 273 in 1-indexed
# Let's check both
println("Token 272 (0-indexed): \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [272])), "\"")
println("Token 271 (0-indexed): \"", escape_string(Inferno.Tokenizer.decode(tokenizer, [271])), "\"")

# Check what the model's vocab size is
println("\nVocab size from tokenizer: ", length(tokenizer.id_to_token))
println("Embedding shape: (1024, 248320)")

# The model has vocab_size = 248320, which includes special tokens
# Let's check the special tokens
println("\nSpecial tokens:")
println("  EOS: ", tokenizer.eos_id)
println("  BOS: ", tokenizer.bos_id)
