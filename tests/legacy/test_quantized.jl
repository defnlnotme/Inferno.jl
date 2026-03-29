using Inferno

println("Loading model with quantized MLP weights...")
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)

# Test generation
prompt = "Hello, my name is"
println("\nPrompt: ", prompt)
println("Generating...")

# Tokenize
input_ids = Inferno.Tokenizer.encode(tok, prompt)
println("Input tokens: ", input_ids)

# Initialize caches
caches = [ModelCPU.init_kv_cache_cpu(model.config, 100) for _ in model.layers]

# Generate - need to use correct signature
output = ModelCPU.generate_cpu(model, Int.(input_ids), 20, caches; temperature=0.7f0)

# output is (next_token, probs) tuple
println("\nNext token: ", output[1])
println("Probabilities shape: ", size(output[2]))

# Decode just the next token
next_tokens = vcat(Int.(input_ids), [output[1]])
output_text = Inferno.Tokenizer.decode(tok, next_tokens)
println("Text so far: ", output_text)
