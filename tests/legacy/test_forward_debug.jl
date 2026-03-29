using Inferno
using Printf
using Inferno.Tokenizer
using Inferno.ModelCPU
using Statistics

println("Loading model...")
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Tokenize prompt
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
tokens = encode(tok, prompt)
println("Tokens: ", tokens)
println("Token strings: ", [tok.id_to_token[t] for t in tokens])

# Get the embedding for the first token
first_tok = tokens[1]
x = model.embed[:, first_tok]

println("\nFirst token embedding:")
println("  Shape: ", size(x))
println("  Stats: min=$(minimum(x)), max=$(maximum(x)), mean=$(mean(x)), std=$(std(x))")

# Check if embedding looks reasonable
println("\n  First 10 values: ", x[1:10])

# Run through the first layer
layer1 = model.layers[1]
caches = [ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:length(model.layers)]

# Apply layer 1
x1 = layer1(x, 0, model.rope, caches[1])
println("\nAfter layer 1:")
println("  Shape: ", size(x1))
println("  Stats: min=$(minimum(x1)), max=$(maximum(x1)), mean=$(mean(x1)), std=$(std(x1))")
println("  First 10 values: ", x1[1:10])

# Run full forward pass
all_logits = ModelCPU.forward_cpu!(model, tokens, 0, caches)
last_logits = all_logits[:, end]

println("\nFinal logits (last position):")
println("  Shape: ", size(last_logits))
println("  Stats: min=$(minimum(last_logits)), max=$(maximum(last_logits)), mean=$(mean(last_logits))")

# Top predictions
sorted_indices = sortperm(last_logits, rev=true)
println("\nTop 10 predictions:")
for i in 1:10
 idx = sorted_indices[i]
 println(" $i: id=$idx logit=$(last_logits[idx]) token=\"$(tok.id_to_token[idx])\"")
end
