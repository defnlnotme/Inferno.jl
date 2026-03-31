using Inferno
using LinearAlgebra

model, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("=== LM Head Check ===")
println("lm_head shape: ", size(model.lm_head))
println("Expected: (vocab_size, hidden_size) = (248320, 1024)")

# Check if lm_head needs transpose
# For matmul: logits = lm_head * x
# If lm_head is (vocab, hidden) and x is (hidden,), then logits is (vocab,)
# This is correct!

println("\n=== Test LM Head MatMul ===")
x_test = randn(Float32, 1024)
logits_test = model.lm_head * x_test
println("x_test norm: ", round(norm(x_test), digits=5))
println("logits_test shape: ", size(logits_test))
println("logits_test norm: ", round(norm(logits_test), digits=5))

# Check lm_head weights
println("\n=== LM Head Weights ===")
println("lm_head norm: ", round(norm(model.lm_head), digits=5))
println("lm_head row 762 (for 'The') norm: ", round(norm(model.lm_head[762, :]), digits=5))
