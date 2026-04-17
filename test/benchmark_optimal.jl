using Inferno
using LinearAlgebra
using Statistics

function benchmark_optimal_config()
    GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    println("=== Optimal Thread Configuration Test ===")
    println("Julia threads: $(Threads.nthreads())")
    println()
    
    prompt = "The capital of France is"
    prompt_tokens = Inferno.Tokenizer.encode(tok, prompt)
    
    # Test different BLAS thread counts with full generation
    for blas_threads in [4, 6, 8, 10, 12]
        BLAS.set_num_threads(blas_threads)
        
        # Warmup
        for _ in 1:3
            Inferno.ModelCPU.reset_states_cpu!(model)
            caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
            logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
        end
        
        # Timed generation
        n_tokens = 20
        
        Inferno.ModelCPU.reset_states_cpu!(model)
        caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
        
        logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
        token = argmax(logits)[1]
        pos = length(prompt_tokens)
        
        times = Float64[]
        
        for _ in 1:n_tokens
            t0 = time()
            logits = Inferno.ModelCPU.forward_cpu!(model, [token], pos, caches; full_logits=false)
            token = argmax(logits)[1]
            pos += 1
            t1 = time()
            push!(times, t1 - t0)
        end
        
        steady_times = times[4:end]
        avg_time = mean(steady_times)
        
        println("BLAS=$blas_threads: $(round(avg_time * 1000, digits=2)) ms/token = $(round(1/avg_time, digits=1)) tok/s")
    end
    
    println()
    println("Best config appears to be:")
    println("  Julia threads: $(Threads.nthreads()) (for chunked partitioning)")
    println("  BLAS threads: 8 (peak for matrix operations)")
    
    # Restore
    BLAS.set_num_threads(10)
end

benchmark_optimal_config()
