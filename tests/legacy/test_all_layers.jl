using Inferno
using Statistics

println("Loading models...")
model_q, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

input_id = 9420  # "Hello"
x_q = copy(model_q.embed[:, input_id+1])
x_f = copy(model_f.embed[:, input_id+1])

println("\nProcessing through all layers...")

for (i, (layer_q, layer_f)) in enumerate(zip(model_q.layers, model_f.layers))
 # Initialize fresh cache for each layer
 cache_q = ModelCPU.init_kv_cache_cpu(model_q.config, 10)
 cache_f = ModelCPU.init_kv_cache_cpu(model_f.config, 10)
 
 # Apply layer
 global x_q, x_f
 
 x_q = layer_q(x_q, 0, model_q.rope, cache_q)
 x_f = layer_f(x_f, 0, model_f.rope, cache_f)
 
 diff = maximum(abs.(x_q .- x_f))
 rel_err = mean(abs.(x_q .- x_f)) / mean(abs.(x_f)) * 100
 
 println("Layer $i ($(layer_q.is_ssm ? "SSM" : "Attn")): max_diff=$diff, rel_err=$rel_err%")
 
 # If error is too large, show details
 if rel_err > 0.01
 println("  WARNING: Large error detected!")
 println("  x_q [1:5]: ", x_q[1:5])
 println("  x_f [1:5]: ", x_f[1:5])
 break
 end
end

# Final norm
println("\nApplying final norm...")
x_final_q = model_q.final_norm(x_q)
x_final_f = model_f.final_norm(x_f)
println("  Max diff: ", maximum(abs.(x_final_q .- x_final_f)))

# LM head
println("\nApplying LM head...")
logits_q = model_q.lm_head * x_final_q
logits_f = model_f.lm_head * x_final_f
println("  Logits shape: ", size(logits_q))
println("  Max diff: ", maximum(abs.(logits_q .- logits_f)))
println("  Relative error: ", mean(abs.(logits_q .- logits_f)) / mean(abs.(logits_f)) * 100, " %")

# Predicted tokens
pred_q = argmax(logits_q)
pred_f = argmax(logits_f)
println("\nPredicted token (quantized): $pred_q = '", Inferno.Tokenizer.decode(tok, [pred_q]), "'")
println("Predicted token (float): $pred_f = '", Inferno.Tokenizer.decode(tok, [pred_f]), "'")
