using Inferno
using LinearAlgebra

function run_profile()
    GGUF_PATH = "tests/models/Qwen3.5-0.8B/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    prompt_tokens = Inferno.Tokenizer.encode(tok, "test")
    
    # Warm up
    Inferno.ModelCPU.reset_states_cpu!(model)
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    _ = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
    
    println("=== Layer-by-Layer Profiling ===")
    println()
    
    Inferno.ModelCPU.reset_states_cpu!(model)
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    
    # Get embedding
    h = model.embed[:, prompt_tokens[end]]
    pos = length(prompt_tokens)
    
    layer_times = Float64[]
    layer_types = String[]
    
    for (i, layer) in enumerate(model.layers)
        t0 = time()
        h = layer(h, pos, model.rope, caches[i])
        t1 = time()
        push!(layer_times, t1 - t0)
        
        op_type = layer.is_ssm ? "SSM" : "Attn"
        push!(layer_types, op_type)
    end
    
    # Final norm + lm_head
    t0 = time()
    Inferno.ModelCPU.rmsnorm_cpu!(model.final_norm_buf, h, model.final_norm)
    mul!(model.lm_head_buf, model.lm_head, model.final_norm_buf)
    t1 = time()
    final_time = t1 - t0
    
    println("Layer times (ms):")
    for i in 1:length(layer_times)
        println("  Layer $i ($(layer_types[i])): $(round(layer_times[i] * 1000, digits=2)) ms")
    end
    println()
    println("Final norm + lm_head: $(round(final_time * 1000, digits=2)) ms")
    println()
    
    # Summary by type
    ssm_time = sum(layer_times[i] for i in 1:length(layer_times) if layer_types[i] == "SSM")
    attn_time = sum(layer_times[i] for i in 1:length(layer_times) if layer_types[i] == "Attn")
    total_layer_time = sum(layer_times)
    
    println("Summary:")
    println("  Total SSM layers: $(round(ssm_time * 1000, digits=1)) ms")
    println("  Total Attn layers: $(round(attn_time * 1000, digits=1)) ms")
    println("  Total layers: $(round(total_layer_time * 1000, digits=1)) ms")
    println("  Final projection: $(round(final_time * 1000, digits=1)) ms")
end

run_profile()
