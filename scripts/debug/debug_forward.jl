using Inferno
using Statistics

println("Loading model...")
model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
println("Model loaded.")

# Initialize
caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
ModelCPU.reset_states_cpu!(model)

# Full forward pass
let hidden = copy(view(model.embed, :, 1))
    for (i, layer) in enumerate(model.layers)
        hidden = layer(hidden, 0, model.rope, caches[i])
    end
    hidden = model.final_norm(hidden)
    global logits = model.lm_head * hidden
end

println("\n=== Forward Pass Results ===")
println("logits shape: ", size(logits))
println("any NaN: ", any(isnan, logits))
println("mean: ", mean(logits))
println("std: ", std(logits))
println("min: ", minimum(logits))
println("max: ", maximum(logits))
top5_idx = sortperm(logits, rev=true)[1:5]
println("top 5 tokens: ", top5_idx)
println("top 5 values: ", logits[top5_idx])
