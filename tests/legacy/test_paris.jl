using Inferno
using Printf
using Inferno.Tokenizer

println("Loading model...")
_, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check for Paris-related tokens
println("Searching for 'Paris' tokens...")
for (token, id) in tok.token_to_id
 if occursin("Paris", token) || occursin("paris", token)
 println(" \"$token\" -> id=$id")
 end
end

# Check for common city tokens
println("\nSearching for city tokens...")
cities = ["Paris", "London", "Berlin", "Rome", "Madrid", "Lyon", "Marseille"]
for city in cities
 id = get(tok.token_to_id, city, nothing)
 if id !== nothing
 println(" \"$city\" -> id=$id")
 end
 # Also check with space prefix
 id_sp = get(tok.token_to_id, " $city", nothing)
 if id_sp !== nothing
 println(" \" $city\" -> id=$id_sp")
 end
end

# Check the token for "Paris" with ID 57591
println("\nToken at id=57591: \"$(tok.id_to_token[57591])\"")

# Check logit for Paris token during inference
using Inferno.ModelCPU
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
tokens = encode(tok, "The capital of France is")
caches = [ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:length(model.layers)]
logits = ModelCPU.forward_cpu!(model, tokens, 0, caches)
last_logits = logits[:, end]

paris_id = 57591
println("\nLogit for 'Paris' (id=$paris_id): $(last_logits[paris_id])")

# Find rank of Paris token
sorted_indices = sortperm(last_logits, rev=true)
paris_rank = findfirst(==(paris_id), sorted_indices)
println("Rank of 'Paris' token: $paris_rank (out of $(length(last_logits)))")
