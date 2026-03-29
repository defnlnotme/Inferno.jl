using Inferno

println("Loading model with quantized MLP weights...")
model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)

# Compare outputs between quantized and non-quantized
println("\nComparing output quality...")

# Test prompt
prompt = "The capital of France is"
input_ids = Inferno.Tokenizer.encode(tok, prompt)

# Run with quantized weights
caches_quant = [ModelCPU.init_kv_cache_cpu(model.config, 50) for _ in model.layers]
output_quant = ModelCPU.forward_cpu!(model, Int.(input_ids), 0, caches_quant)

println("\nQuantized model logits (first 10):")
println(output_quant[1:10, end])

# Load non-quantized model for comparison
println("\nLoading model without quantization...")
model_full, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

caches_full = [ModelCPU.init_kv_cache_cpu(model_full.config, 50) for _ in model_full.layers]
output_full = ModelCPU.forward_cpu!(model_full, Int.(input_ids), 0, caches_full)

println("\nFull dequantized model logits (first 10):")
println(output_full[1:10, end])

# Compare
diff = output_quant[:, end] .- output_full[:, end]
println("\nLogit difference statistics:")
println("  Max absolute diff: ", maximum(abs.(diff)))
println("  Mean absolute diff: ", sum(abs.(diff)) / length(diff))
println("  Correlation: ", cor(output_quant[:, end], output_full[:, end]))

# Test generation
println("\n--- Generation Test ---")
println("Prompt: ", prompt)

# Generate with quantized
gen_tokens = Int.(input_ids)
for i in 1:10
    cache = ModelCPU.init_kv_cache_cpu(model.config, 50)
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 50) for _ in model.layers]
    
    # Re-run from start each time (simplistic approach)
    logits = ModelCPU.forward_cpu!(model, gen_tokens, 0, caches)
    next_token = argmax(logits[:, end])
    push!(gen_tokens, next_token)
end

output_text = Inferno.Tokenizer.decode(tok, gen_tokens)
println("Generated (quantized): ", output_text)
