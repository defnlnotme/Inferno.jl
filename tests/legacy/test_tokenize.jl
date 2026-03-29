using Inferno
using Printf
using Inferno.Tokenizer

println("Loading model...")
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check tokenizer
println("\nTokenizer type: ", typeof(tok))
println("EOS ID: ", tok.eos_id)
println("BOS ID: ", tok.bos_id)

# Check special tokens
println("\nSpecial tokens:")
for (k, v) in tok.special_tokens
 println(" $k -> $v")
end

# Try simple tokenization
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
println("\nPrompt: \"$prompt\"")

# Tokenize
tokens = encode(tok, prompt)
println("Tokens: ", tokens)

# Check what tokens decode to
println("\nDecoded tokens:")
for t in tokens[1:min(10, length(tokens))]
 println(" $t -> \"$(tok.id_to_token[t])\"")
end
