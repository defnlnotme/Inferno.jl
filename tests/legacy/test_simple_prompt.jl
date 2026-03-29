using Inferno
using Printf
using Inferno.Tokenizer
using Inferno.ModelCPU
using Statistics

println("Loading model...")
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Try a simple prompt without chat template
simple_prompt = "The capital of France is"
tokens = encode(tok, simple_prompt)
println("Simple prompt: \"$simple_prompt\"")
println("Tokens: ", tokens)
println("Token strings: ", [tok.id_to_token[t] for t in tokens])

# Create caches and run forward pass
caches = [ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:length(model.layers)]
logits = ModelCPU.forward_cpu!(model, tokens, 0, caches)

# Check predictions
last_logits = logits[:, end]
sorted_indices = sortperm(last_logits, rev=true)

println("\nTop 10 predictions for simple prompt:")
for i in 1:10
 idx = sorted_indices[i]
 println(" $i: id=$idx logit=$(last_logits[idx]) token=\"$(tok.id_to_token[idx])\"")
end

# Apply softmax
exp_logits = exp.(last_logits .- maximum(last_logits))
probs = exp_logits ./ sum(exp_logits)
sorted_prob = sortperm(probs, rev=true)

println("\nTop 10 tokens by probability:")
for i in 1:10
 idx = sorted_prob[i]
 println(" $i: id=$idx prob=$(probs[idx]*100)% token=\"$(tok.id_to_token[idx])\"")
end

# Generate a few tokens greedily
println("\nGenerating 5 tokens greedily:")
generated = Int[]
current_tokens = copy(tokens)
for i in 1:5
 # Reset caches for each generation
 caches = [ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:length(model.layers)]
 logits = ModelCPU.forward_cpu!(model, current_tokens, 0, caches)
 next_token = argmax(logits[:, end])
 push!(generated, next_token)
 push!(current_tokens, next_token)
 println(" Generated token $i: id=$next_token \"$(tok.id_to_token[next_token])\"")
end

println("\nFull generated text: \"$(tok.id_to_token[tokens[1]])$(join([tok.id_to_token[t] for t in tokens[2:end]]))$(join([tok.id_to_token[t] for t in generated]))\"")
