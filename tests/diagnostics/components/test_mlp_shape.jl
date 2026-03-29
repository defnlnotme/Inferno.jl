#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Extract raw tensor and check shape
gate_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "blk.0.ffn_gate.weight")
gate_transposed = gate_raw'

println("Raw gate weight shape: ", size(gate_raw))
println("Transposed gate weight shape: ", size(gate_transposed))

# According to MLPCPU, gate_weight should be (intermediate, hidden) = (3584, 1024)
# so that gate = gate_weight * x works where x is (hidden,) = (1024,)
println("\nExpected shape for MLP: (3584, 1024)")
println("Actual shape after transpose: ", size(gate_transposed))

# Check the model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
mlp = model.layers[1].mlp

println("\nLoaded MLP gate_weight shape: ", size(mlp.gate_weight))
println("Loaded MLP up_weight shape: ", size(mlp.up_weight))
println("Loaded MLP down_weight shape: ", size(mlp.down_weight))

# Verify computation
x = randn(Float32, 1024)
gate_out = mlp.gate_weight * x
println("\nTest: gate_weight * x shape: ", size(gate_out), " (expected: (3584,))")
