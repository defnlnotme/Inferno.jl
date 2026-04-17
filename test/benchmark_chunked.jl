using Inferno
using LinearAlgebra
using Statistics

"""
Chunked lm_head projection with configurable chunk count.
This is similar to llama.cpp's parallel row distribution.
"""
function lm_head_project_chunked!(output::Vector{Float32}, weight::Matrix{Float32}, 
                                   hidden::Vector{Float32}; nchunks::Int=4)
    vocab_size = size(weight, 1)
    chunk_size = cld(vocab_size, nchunks)
    
    @sync for chunk in 1:nchunks
        i_start = (chunk - 1) * chunk_size + 1
        i_end = min(chunk * chunk_size, vocab_size)
        
        Threads.@spawn begin
            mul!(view(output, i_start:i_end), view(weight, i_start:i_end, :), hidden)
        end
    end
end

function benchmark_with_chunked_lmhead()
    GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    println("=== Full Model with Chunked lm_head ===")
    println()
    
    # Use optimal BLAS config
    BLAS.set_num_threads(8)
    println("BLAS threads: 8 (optimal)")
    println("Julia threads: $(Threads.nthreads())")
    println()
    
    prompt = "The capital of France is"
    prompt_tokens = Inferno.Tokenizer.encode(tok, prompt)
    
    # Warmup
    for _ in 1:3
        Inferno.ModelCPU.reset_states_cpu!(model)
        caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
        _ = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
    end
    
    # Test standard vs chunked lm_head
    println("Testing lm_head variants:")
    
    hidden = randn(Float32, 1024) * 0.01f0
    output = Vector{Float32}(undef, 248320)
    
    # Standard BLAS
    t0 = time()
    for _ in 1:100
        mul!(output, model.lm_head, hidden)
    end
    t1 = time()
    blas_time = (t1 - t0) / 100
    println("  Standard BLAS: $(round(blas_time * 1000, digits=2)) ms")
    
    # Chunked
    for nchunks in [2, 4, 8]
        t0 = time()
        for _ in 1:100
            lm_head_project_chunked!(output, model.lm_head, hidden; nchunks=nchunks)
        end
        t1 = time()
        chunked_time = (t1 - t0) / 100
        println("  Chunked ($nchunks): $(round(chunked_time * 1000, digits=2)) ms, speedup: $(round(blas_time/chunked_time, digits=2))x")
    end
    println()
    
    println("Full generation benchmark (30 tokens):")
    println()
    
    n_tokens = 30
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
    
    println("Result: $(round(avg_time * 1000, digits=2)) ms/token = $(round(1/avg_time, digits=1)) tok/s")
end

benchmark_with_chunked_lmhead()
