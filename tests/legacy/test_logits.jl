using Inferno
using Printf
using Inferno.Tokenizer
using Inferno.ModelCPU

println("Loading model...")
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Tokenize prompt
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
tokens = encode(tok, prompt)
println("Tokens: ", tokens)

# Create KV caches
caches = [ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:length(model.layers)]

# Run forward pass
println("\nRunning forward pass...")
logits = ModelCPU.forward_cpu!(model, tokens, 0, caches)

println("Logits shape: ", size(logits))

# Check the last token's logits
last_logits = logits[:, end]

# Find top 10 tokens
sorted_indices = sortperm(last_logits, rev=true)
println("\nTop 10 predicted tokens:")
for i in 1:10
 idx = sorted_indices[i]
 token_str = get(tok.id_to_token, idx, "UNKNOWN")
 println(" $i: token_id=$idx logit=$(last_logits[idx]) token=\"$token_str\"")
end

# Apply softmax to get probabilities
exp_logits = exp.(last_logits .- maximum(last_logits))
probs = exp_logits ./ sum(exp_logits)

# Find top tokens by probability
sorted_prob_indices = sortperm(probs, rev=true)
println("\nTop 10 tokens by probability:")
for i in 1:10
 idx = sorted_prob_indices[i]
 token_str = get(tok.id_to_token, idx, "UNKNOWN")
 println(" $i: token_id=$idx prob=$(probs[idx]*100)% token=\"$token_str\"")
end

# Also check what "Hello" would be (expected response)
hello_token = get(tok.token_to_id, "Hello", nothing)
if hello_token !== nothing
 println("\n\"Hello\" token ID: $hello_token")
 println(" \"Hello\" logit: $(last_logits[hello_token+1])")
 println(" \"Hello\" prob: $(probs[hello_token+1]*100)%")
end

# Check for common response tokens
response_tokens = ["Hi", "Hello", "Hey", "I", "The", "Hello", " there", "!"]
println("\nCommon response tokens:")
for t in response_tokens
 id = get(tok.token_to_id, t, nothing)
 if id !== nothing
 println(" \"$t\" -> id=$id, logit=$(last_logits[id+1]), prob=$(probs[id+1]*100)%")
 end
end
