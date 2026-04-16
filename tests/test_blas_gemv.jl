using LinearAlgebra
using Base.Threads

vocab_size = 248320
hidden_size = 1024

lm_head = randn(Float32, vocab_size, hidden_size)
hidden = randn(Float32, hidden_size)
output = similar(hidden, vocab_size)

# Using BLAS.gemv! with correct signature (Char, not String)
times = Float64[]
for i in 1:20
    t0 = time()
    BLAS.gemv!('N', 1.0f0, lm_head, hidden, 0.0f0, output)
    t1 = time()
    push!(times, t1 - t0)
end
println("BLAS.gemv!: ", round(sum(times)/length(times)*1000, digits=2), " ms")
