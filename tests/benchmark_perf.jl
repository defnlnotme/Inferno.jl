using Inferno
using LinearAlgebra

function run_benchmark()
    model, tok = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B/")
    
    println("="^60)
    println("Inferno.jl CPU Inference Benchmark")
    println("="^60)
    println()
    
    # Test different prompts
    prompts = [
        ("Math", "What is 2 + 2 ?"),
        ("Knowledge", "The capital of France is"),
        ("Code", "function fibonacci(n)"),
    ]
    
    results = []
    
    for (name, prompt) in prompts
        println("--- $name ---")
        println("Prompt: \"$prompt\"")
        
        # Warm up
        Inferno.stream_to_stdout_cpu(model, tok, prompt;
            max_tokens=20, temperature=0.0f0, show_tps=false)
        println()
        
        # Measure
        times = Float64[]
        tokens_generated = 0
        
        for i in 1:3
            Inferno.ModelCPU.reset_states_cpu!(model)
            prompt_tokens = Inferno.Tokenizer.encode(tok, prompt)
            caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
            
            # Process prompt
            logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
            pos = length(prompt_tokens)
            
            # Generate tokens
            start_time = time()
            for j in 1:30
                token = argmax(vec(logits))
                logits = Inferno.ModelCPU.forward_cpu!(model, [token], pos, caches; full_logits=false)
                pos += 1
            end
            end_time = time()
            
            push!(times, end_time - start_time)
        end
        
        avg_time = sum(times) / length(times)
        tps = 30 / avg_time
        ms_per_token = avg_time * 1000 / 30
        
        println("  30 tokens in $(round(avg_time, digits=2))s")
        println("  $(round(tps, digits=1)) tokens/sec")
        println("  $(round(ms_per_token, digits=1)) ms/token")
        println()
        
        push!(results, (name, tps, ms_per_token))
    end
    
    println("="^60)
    println("Summary")
    println("="^60)
    avg_tps = sum(r[2] for r in results) / length(results)
    avg_ms = sum(r[3] for r in results) / length(results)
    println("Average: $(round(avg_tps, digits=1)) tok/s, $(round(avg_ms, digits=1)) ms/token")
    println()
    
    # Memory allocation check
    println("--- Memory Allocation ---")
    Inferno.ModelCPU.reset_states_cpu!(model)
    prompt_tokens = Inferno.Tokenizer.encode(tok, "test")
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    
    logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
    
    # Single token allocation
    allocs = @allocated begin
        logits = Inferno.ModelCPU.forward_cpu!(model, [1], length(prompt_tokens), caches; full_logits=false)
    end
    println("Allocations per token: $allocs bytes")
end

run_benchmark()
