using Inferno

# Compare memory usage between quantized and non-quantized loading

println("=" ^ 60)
println("Memory Comparison: Quantized vs Non-Quantized")
println("=" ^ 60)

# Force GC before tests
GC.gc()

# Test 1: Load with quantized weights
println("\n1. Loading with quantized MLP weights...")
model_quant, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)

# Get memory stats
mem_quant = Base.gc_live_bytes()
println("   Memory after loading: $(mem_quant / 1024 / 1024) MB")

# Show weight types
l1 = model_quant.layers[1]
println("\n   Layer 1 MLP weight types:")
println("   - gate: ", typeof(l1.mlp.gate_weight))
println("   - up: ", typeof(l1.mlp.up_weight))
println("   - down: ", typeof(l1.mlp.down_weight))

# Clear model
model_quant = nothing
GC.gc()

# Test 2: Load without quantized weights (fully dequantized)
println("\n2. Loading with full dequantization...")
model_full, tok = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

mem_full = Base.gc_live_bytes()
println("   Memory after loading: $(mem_full / 1024 / 1024) MB")

# Show weight types
l1 = model_full.layers[1]
println("\n   Layer 1 MLP weight types:")
println("   - gate: ", typeof(l1.mlp.gate_weight))
println("   - up: ", typeof(l1.mlp.up_weight))
println("   - down: ", typeof(l1.mlp.down_weight))

# Summary
println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Quantized MLP weights: $(mem_quant / 1024 / 1024) MB")
println("Full dequantized:     $(mem_full / 1024 / 1024) MB")
println("Memory savings:       $((1 - mem_quant / mem_full) * 100) %")
