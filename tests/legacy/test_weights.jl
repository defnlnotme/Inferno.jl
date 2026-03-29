using Inferno
using Printf
using Statistics

println("Loading model in both quantized and float modes...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Compare embeddings
println("\nEmbedding comparison:")
println(" Quantized embedding shape: ", size(model_q.embed))
println(" Float embedding shape: ", size(model_f.embed))

# Check a specific token embedding
tok_id = 1 # "!" token
emb_q = model_q.embed[:, tok_id]
emb_f = model_f.embed[:, tok_id]

println("\nToken $tok_id embedding comparison:")
println(" Quantized [1:5]: ", emb_q[1:5])
println(" Float [1:5]: ", emb_f[1:5])
println(" Max diff: ", maximum(abs.(emb_q .- emb_f)))

# Check LM head
println("\nLM head comparison:")
println(" Quantized LM head type: ", typeof(model_q.lm_head))
println(" Float LM head type: ", typeof(model_f.lm_head))

# Compare a row of LM head
println("\nLM head row 1 comparison:")
row_q = model_q.lm_head[1, :]
row_f = model_f.lm_head[1, :]
println(" Quantized [1:5]: ", row_q[1:5])
println(" Float [1:5]: ", row_f[1:5])
println(" Max diff: ", maximum(abs.(row_q .- row_f)))

# Check if LM head is quantized
if isa(model_q.lm_head, Inferno.QuantsCPU.Q4_K_Matrix)
 println("\nLM head is Q4_K quantized")
 println(" inner_dim: ", model_q.lm_head.inner_dim)
 println(" outer_dim: ", model_q.lm_head.outer_dim)
 println(" num_blocks: ", model_q.lm_head.num_blocks)
end
