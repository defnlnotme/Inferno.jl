using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Reset states
Inferno.ModelCPU.reset_states_cpu!(model)

# Get "The" embedding
token = 761
x = model.embed[:, token + 1]

# Run through layer 1
layer1 = model.layers[1]
ssm = layer1.op
cache1 = Inferno.ModelCPU.init_kv_cache_cpu(model.config)

# Forward through layer 1
x_out = layer1(x, 0, model.rope, cache1)

println("=== Layer 1 Output ===")
println("x_out norm: ", round(norm(x_out), digits=5))

# Check SSM state after forward
println("\nSSM state after forward:")
println("h norm: ", round(norm(ssm.h), digits=5))
println("h[:,:,1] norm: ", round(norm(ssm.h[:,:,1]), digits=5))

# Now run through all layers
Inferno.ModelCPU.reset_states_cpu!(model)
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

for (i, layer) in enumerate(model.layers)
    global x
    x = layer(x, i-1, model.rope, caches[i])
    if i <= 5
        println("After layer $i: norm = ", round(norm(x), digits=5))
    end
end

# Final norm
x_final = model.final_norm(x)
logits = model.lm_head * x_final

println("\n=== Final ===")
println("x_final norm: ", round(norm(x_final), digits=5))
println("logits norm: ", round(norm(logits), digits=5))

# Top tokens
top_idx = argmax(logits)
println("\nTop token: ", top_idx-1, " -> '", Inferno.Tokenizer.decode(tok, [top_idx-1]), "'")
