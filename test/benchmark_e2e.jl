using Inferno
using LinearAlgebra
using Statistics

function benchmark_full_generation()
    GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    println("=== Full Generation Benchmark ===")
    println("Prompt: 'The capital of France is'")
    println()
    
    prompt = "The capital of France is"
    prompt_tokens = Inferno.Tokenizer.encode(tok, prompt)
    println("Prompt tokens: $prompt_tokens ($(length(prompt_tokens)) tokens)")
    println()
    
    println("Settings:")
    println("  Julia threads: ", Threads.nthreads())
    println("  BLAS threads: ", BLAS.get_num_threads())
    println()
    
    # Warmup
    println("Warming up...")
    for _ in 1:3
        Inferno.ModelCPU.reset_states_cpu!(model)
        caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
        logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
    end
    println("Warmup done.")
    println()
    
    # Benchmark generation
    n_tokens = 30
    println("Generating $n_tokens tokens...")
    println()
    
    Inferno.ModelCPU.reset_states_cpu!(model)
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    
    # Initial forward for prompt
    t0 = time()
    logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
    prompt_time = time() - t0
    println("Prompt processing ($(length(prompt_tokens)) tokens): $(round(prompt_time * 1000, digits=1)) ms")
    
    # Generate tokens one at a time
    token = argmax(logits)[1]  # Get the index as Int
    pos = length(prompt_tokens)
    
    times = Float64[]
    tokens_generated = Int[]
    
    for i in 1:n_tokens
        t0 = time()
        
        logits = Inferno.ModelCPU.forward_cpu!(model, [token], pos, caches; full_logits=false)
        token = argmax(logits)[1]
        pos += 1
        
        t1 = time()
        push!(times, t1 - t0)
        push!(tokens_generated, token)
    end
    
    # Calculate stats
    warmup_tokens = 3
    steady_times = times[warmup_tokens+1:end]
    avg_time = mean(steady_times)
    med_time = median(steady_times)
    min_time = minimum(steady_times)
    
    println()
    println("=== Results (excluding first $warmup_tokens tokens) ===")
    println("  Mean time: $(round(avg_time * 1000, digits=2)) ms/token")
    println("  Median time: $(round(med_time * 1000, digits=2)) ms/token")
    println("  Min time: $(round(min_time * 1000, digits=2)) ms/token")
    println("  Throughput: $(round(1/avg_time, digits=1)) tok/s")
    println()
    
    # Decode output
    output_text = Inferno.Tokenizer.decode(tok, vcat(prompt_tokens, tokens_generated))
    println("Generated text:")
    println(output_text)
end

benchmark_full_generation()

# Test BLAS thread scaling
println()
println("=== BLAS Thread Scaling (layer-only, no lm_head) ===")
println()

GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
model, tok = Inferno.load_model_cpu(GGUF_PATH)
prompt_tokens = Inferno.Tokenizer.encode(tok, "test")

for nt in [1, 2, 4, 8, 10, 16, 20]
    BLAS.set_num_threads(nt)
    
    # Quick test - one token through all layers
    n = 20
    times = Float64[]
    
    for _ in 1:n
        Inferno.ModelCPU.reset_states_cpu!(model)
        caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
        
        t0 = time()
        h = model.embed[:, prompt_tokens[end]]
        for i in 1:24
            h = model.layers[i](h, length(prompt_tokens), model.rope, caches[i])
        end
        Inferno.ModelCPU.rmsnorm_cpu!(model.final_norm_buf, h, model.final_norm)
        # Skip lm_head for this test
        t1 = time()
        push!(times, t1 - t0)
    end
    
    per_token = mean(times)
    println("BLAS=$nt threads: $(round(per_token * 1000, digits=2)) ms (layers only)")
end

BLAS.set_num_threads(10)
