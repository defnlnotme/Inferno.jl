using Inferno
using LinearAlgebra

model, tok = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B/")

prompt_tokens = Inferno.Tokenizer.encode(tok, "test")

# Initialize
Inferno.ModelCPU.reset_states_cpu!(model)
caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]

# Process prompt
logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)

# Warm up more
for i in 1:10
    Inferno.ModelCPU.forward_cpu!(model, [1], length(prompt_tokens), caches; full_logits=false)
end

println("=== Detailed Allocation Analysis ===")
println()

# Single token generation
allocs = @allocated Inferno.ModelCPU.forward_cpu!(model, [1], length(prompt_tokens), caches; full_logits=false)
println("Single token forward: $allocs bytes")

# Check layer by layer
h = model.embed[:, 1]
layer_allocs = Int[]
for i in 1:length(model.layers)
    a = @allocated model.layers[i](h, length(prompt_tokens), model.rope, caches[i])
    push!(layer_allocs, a)
end

println("\nLayer allocations:")
for i in 1:length(model.layers)
    println("  Layer $i: $(layer_allocs[i]) bytes")
end

total = sum(layer_allocs)
println("\nTotal layer allocations: $total bytes")

# lm_head
allocs = @allocated Inferno.ModelCPU.lm_head_project!(model.lm_head_buf, model.lm_head, model.final_norm_buf)
println("lm_head allocations: $allocs bytes")
