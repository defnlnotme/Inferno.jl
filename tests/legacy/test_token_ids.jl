using Inferno
using Printf
using Inferno.Tokenizer
using Inferno.ModelCPU

println("Loading model...")
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check id_to_token mapping - it's 1-indexed!
println("\nChecking token ID mapping (1-indexed):")

# Check some known tokens
test_ids = [1, 2, 11, 101, 1001, 9421, 248047, 248048]
for id in test_ids
 if id <= length(tok.id_to_token)
 println(" id=$id -> \"$(tok.id_to_token[id])\"")
 else
 println(" id=$id -> OUT OF RANGE (max=$(length(tok.id_to_token)))")
 end
end

# Check the top predicted tokens from logits
caches = [ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:length(model.layers)]
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
tokens = encode(tok, prompt)
logits = ModelCPU.forward_cpu!(model, tokens, 0, caches)
last_logits = logits[:, end]

sorted_indices = sortperm(last_logits, rev=true)
println("\nTop 5 token IDs from logits (model output is 0-indexed):")
for i in 1:5
 idx = sorted_indices[i] # This is 1-indexed (Julia array)
 # The model outputs logits with 0-indexed IDs
 # So we need to convert: model_id = idx - 1
 model_id = idx - 1
 println(" Position $i: array_idx=$idx, model_id=$model_id")
 if idx <= length(tok.id_to_token)
 println(" -> \"$(tok.id_to_token[idx])\"")
 end
end

# Check the tokenizer indexing
println("\nTokenizer indexing check:")
println(" \"Hello\" -> token_to_id = $(get(tok.token_to_id, "Hello", "NOT FOUND"))")
# Check if token 9421 is "Hello"
if 9421 <= length(tok.id_to_token)
 println(" id_to_token[9421] = \"$(tok.id_to_token[9421])\"")
end

# Check vocab size
println("\nModel vocab size: ", model.config.vocab_size)
println("Tokenizer id_to_token length: ", length(tok.id_to_token))
