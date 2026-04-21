using Inferno
using Inferno.Gemma4
using Inferno.Gemma4Loader

model, tok = Gemma4Loader.load_gemma4("test/models/gemma-4-E2B-it"; max_seq_len=256)

# Test single token forward pass
token_id = 2  # BOS token
kv_caches = Gemma4.init_kv_cache(model.config, 256)
println("Running forward pass with BOS token...")
hidden = Gemma4.forward!(model, [token_id], 0, kv_caches)
println("Forward pass done! Hidden norm: ", sum(abs2, hidden))
println("Hidden min: ", minimum(hidden), " max: ", maximum(hidden))

# Get logits
logits = Gemma4.get_logits(model, hidden)
println("Logits shape: ", size(logits))
println("Logits min: ", minimum(logits), " max: ", maximum(logits))
println("Top 5 token IDs: ", sortperm(logits, rev=true)[1:5] .- 1)
