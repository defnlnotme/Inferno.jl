using Inferno
using Printf
using Inferno.Tokenizer

println("Loading model...")
_, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check what tokens 272, 221, 14, 12, 319 are
test_ids = [272, 221, 14, 12, 319, 3710, 1711, 9617]
println("\nToken ID to string mapping:")
for id in test_ids
 println(" id=$id -> \"$(tok.id_to_token[id])\"")
end

# Print byte representation
println("\nByte representation:")
for id in test_ids
 token = tok.id_to_token[id]
 bytes = Vector{UInt8}(token)
 println(" id=$id -> $(bytes) -> \"$(escape_string(token))\"")
end

# Check what the expected next tokens should be
# For "The capital of France is", we expect " Paris" or similar
expected = [" Paris", " Paris.", " Paris,", " Paris!", " the"]
println("\nExpected token IDs:")
for t in expected
 id = get(tok.token_to_id, t, nothing)
 if id !== nothing
 println(" \"$t\" -> id=$id")
 else
 println(" \"$t\" -> NOT FOUND")
 end
end

# Also try without space
expected2 = ["Paris", "paris", "Paris.", "Paris, "]
println("\nExpected token IDs (no space prefix):")
for t in expected2
 id = get(tok.token_to_id, t, nothing)
 if id !== nothing
 println(" \"$t\" -> id=$id")
 else
 println(" \"$t\" -> NOT FOUND")
 end
end
