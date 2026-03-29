using Inferno
using Printf
using Statistics
using Inferno.Tokenizer
using Inferno.ModelCPU

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Simple prompt
prompt = "The capital of France is"
tokens = encode(tok, prompt)
println("Tokens: ", tokens)

# Create caches
caches_q = [ModelCPU.init_kv_cache_cpu(model_q.config) for _ in 1:length(model_q.layers)]
caches_f = [ModelCPU.init_kv_cache_cpu(model_f.config) for _ in 1:length(model_f.layers)]

# Forward pass
println("\nRunning forward pass...")
logits_q = ModelCPU.forward_cpu!(model_q, tokens, 0, caches_q)
logits_f = ModelCPU.forward_cpu!(model_f, tokens, 0, caches_f)

println("Logits shape: ", size(logits_q))

# Compare logits at last position
last_q = logits_q[:, end]
last_f = logits_f[:, end]

println("\nLast position logits comparison:")
println(" Quantized [1:5]: ", last_q[1:5])
println(" Float [1:5]: ", last_f[1:5])
println(" Max diff: ", maximum(abs.(last_q .- last_f)))

# Check top predictions
sorted_q = sortperm(last_q, rev=true)
sorted_f = sortperm(last_f, rev=true)

println("\nTop 5 predictions (quantized):")
for i in 1:5
 idx = sorted_q[i]
 println(" $i: id=$idx logit=$(last_q[idx]) token=\"$(tok.id_to_token[idx])\"")
end

println("\nTop 5 predictions (float):")
for i in 1:5
 idx = sorted_f[i]
 println(" $i: id=$idx logit=$(last_f[idx]) token=\"$(tok.id_to_token[idx])\"")
end
