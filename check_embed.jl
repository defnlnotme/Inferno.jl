using Inferno

model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("Embedding shape: ", size(model.embed))
println("LM head shape: ", size(model.lm_head))
println("Final norm weight shape: ", size(model.final_norm.weight))

# Check a sample embedding
println("\nSample embedding for token 761 (\"The\"):")
emb = model.embed[:, 761]
println("  First 5 values: ", emb[1:5])
println("  Norm: ", sum(abs2.(emb)))
