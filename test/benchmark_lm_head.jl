using Inferno
using LinearAlgebra
using Statistics
using LoopVectorization

function benchmark_lm_head()
    GGUF_PATH = "test/models/Qwen3.5-0.8B-GGUF/"
    model, tok = Inferno.load_model_cpu(GGUF_PATH)
    
    println("=== LM Head Benchmark ===")
    println("Vocab size: $(size(model.lm_head, 1))")
    println("Hidden size: $(size(model.lm_head, 2))")
    println("Current BLAS threads: ", BLAS.get_num_threads())
    println()
    
    # Create test input
    hidden = randn(Float32, 1024) * 0.01f0
    output = Vector{Float32}(undef, 248320)
    
    # Test: Standard BLAS mul!
    # Warmup
    for _ in 1:10
        mul!(output, model.lm_head, hidden)
    end
    
    n_iters = 50
    t0 = time()
    for _ in 1:n_iters
        mul!(output, model.lm_head, hidden)
    end
    t1 = time()
    blas_time = (t1 - t0) / n_iters
    
    println("1. BLAS mul! (10 threads): $(round(blas_time * 1000, digits=2)) ms")
    println("   Throughput: $(round(1/blas_time, digits=1)) iters/sec")
    println()
    
    # Test: Single-threaded BLAS
    old_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    
    # Warmup
    for _ in 1:10
        mul!(output, model.lm_head, hidden)
    end
    
    t0 = time()
    for _ in 1:n_iters
        mul!(output, model.lm_head, hidden)
    end
    t1 = time()
    blas1_time = (t1 - t0) / n_iters
    
    println("2. BLAS mul! (1 thread): $(round(blas1_time * 1000, digits=2)) ms")
    println("   Throughput: $(round(1/blas1_time, digits=1)) iters/sec")
    println("   Multi-thread speedup: $(round(blas_time/blas1_time, digits=2))x")
    println()
    
    BLAS.set_num_threads(old_threads)
    
    # Test: Current chunked approach
    function lm_head_chunked!(output, weight, hidden; nchunks=4)
        vocab_size = size(weight, 1)
        chunk_size = cld(vocab_size, nchunks)
        
        tasks = Vector{Task}(undef, nchunks)
        for chunk in 1:nchunks
            i_start = (chunk - 1) * chunk_size + 1
            i_end = min(chunk * chunk_size, vocab_size)
            
            tasks[chunk] = Threads.@spawn begin
                mul!(view(output, i_start:i_end), view(weight, i_start:i_end, :), hidden)
            end
        end
        
        for task in tasks
            wait(task)
        end
    end
    
    # Warmup
    for _ in 1:10
        lm_head_chunked!(output, model.lm_head, hidden; nchunks=4)
    end
    
    t0 = time()
    for _ in 1:n_iters
        lm_head_chunked!(output, model.lm_head, hidden; nchunks=4)
    end
    t1 = time()
    chunked_time = (t1 - t0) / n_iters
    
    println("3. Chunked (4 threads): $(round(chunked_time * 1000, digits=2)) ms")
    println("   Throughput: $(round(1/chunked_time, digits=1)) iters/sec")
    println("   vs BLAS-10: $(round(blas_time/chunked_time, digits=2))x")
    println()
    
    # Test: More chunks with spawn+wait
    for n in [2, 8, 16]
        # Warmup
        for _ in 1:5
            lm_head_chunked!(output, model.lm_head, hidden; nchunks=n)
        end
        
        t0 = time()
        for _ in 1:n_iters
            lm_head_chunked!(output, model.lm_head, hidden; nchunks=n)
        end
        t1 = time()
        t = (t1 - t0) / n_iters
        
        println("   Chunked ($n chunks): $(round(t * 1000, digits=2)) ms")
    end
    println()
    
    # Test: Static work distribution (no dynamic task creation)
    function lm_head_static!(output, weight, hidden)
        nth = Threads.nthreads()
        vocab_size = size(weight, 1)
        rows_per_thread = cld(vocab_size, nth)
        
        Threads.@threads for t in 1:nth
            i_start = (t - 1) * rows_per_thread + 1
            i_end = min(t * rows_per_thread, vocab_size)
            if i_start <= vocab_size
                mul!(view(output, i_start:i_end), view(weight, i_start:i_end, :), hidden)
            end
        end
    end
    
    # Warmup
    for _ in 1:10
        lm_head_static!(output, model.lm_head, hidden)
    end
    
    t0 = time()
    for _ in 1:n_iters
        lm_head_static!(output, model.lm_head, hidden)
    end
    t1 = time()
    static_time = (t1 - t0) / n_iters
    
    println("4. Static @threads: $(round(static_time * 1000, digits=2)) ms")
    println("   Throughput: $(round(1/static_time, digits=1)) iters/sec")
    println("   vs BLAS-10: $(round(blas_time/static_time, digits=2))x")
    println()
    
    println("Summary:")
    println("  BLAS(10t):   $(round(blas_time * 1000, digits=2)) ms")
    println("  BLAS(1t):    $(round(blas1_time * 1000, digits=2)) ms")
    println("  Chunked(4):  $(round(chunked_time * 1000, digits=2)) ms")
    println("  Static(20t): $(round(static_time * 1000, digits=2)) ms")
    
    # Estimate model perf impact
    println()
    println("Full model projection:")
    println("  18 SSM layers @ 0.097ms = $(round(0.097 * 18, digits=1)) ms")
    println("  6 Attn layers @ 1.9ms = $(round(1.9 * 6, digits=1)) ms")
    println("  lm_head @ $(round(blas_time * 1000, digits=2))ms = $(round(blas_time * 1000, digits=2)) ms")
    total_est = 0.097 * 18 + 1.9 * 6 + blas_time * 1000
    println("  Estimated total: $(round(total_est, digits=1)) ms")
    println("  Estimated tok/s: $(round(1000/total_est, digits=1))")
end

benchmark_lm_head()
