using LinearAlgebra
using BenchmarkTools
using Printf

# Mocking parts of ModelCPU
function lm_head_project_current!(output::Vector{Float32}, weight::Matrix{Float32}, hidden::Vector{Float32}; nchunks::Int=4)
    vocab_size = size(weight, 1)
    chunk_size = cld(vocab_size, nchunks)
    
    # Pre-allocate temporary buffers for each chunk
    chunk_outputs = [Vector{Float32}(undef, chunk_size) for _ in 1:nchunks]
    
    # Use spawn+wait instead of @threads to reduce allocations
    tasks = Vector{Task}(undef, nchunks)
    for chunk in 1:nchunks
        i_start = (chunk - 1) * chunk_size + 1
        i_end = min(chunk * chunk_size, vocab_size)
        actual_size = i_end - i_start + 1
        
        tasks[chunk] = Threads.@spawn begin
            buf = view(chunk_outputs[chunk], 1:actual_size)
            BLAS.gemv!('N', 1.0f0, view(weight, i_start:i_end, :), hidden, 0.0f0, buf)
        end
    end
    
    # Wait for all tasks to complete and copy results
    for task in tasks
        wait(task)
    end
    
    # Copy results back to output
    for chunk in 1:nchunks
        i_start = (chunk - 1) * chunk_size + 1
        i_end = min(chunk * chunk_size, vocab_size)
        actual_size = i_end - i_start + 1
        output[i_start:i_end] .= chunk_outputs[chunk][1:actual_size]
    end
end

function benchmark()
    vocab_size = 151936
    hidden_size = 1024
    
    weight = rand(Float32, vocab_size, hidden_size)
    hidden = rand(Float32, hidden_size)
    output = Vector{Float32}(undef, vocab_size)
    
    println("Benchmarking current lm_head_project! (Float32)")
    b = @benchmark lm_head_project_current!($output, $weight, $hidden, nchunks=8)
    display(b)
    println()
    
    # Print allocations
    allocs = b.allocs
    memory = b.memory
    @printf("Allocations: %d, Memory: %.2f KB\n", allocs, memory / 1024)
end

benchmark()
