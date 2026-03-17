#!/usr/bin/env julia

"""
Debug runner for Inferno inference testing.
This script runs comprehensive step-by-step debugging tests.
"""

using Pkg
Pkg.activate(@__DIR__)

# Load the test environment
using Test
using Inferno

println("=== INFERNO DEBUGGING SUITE ===")
println("Running comprehensive inference debugging tests...")
println()

# Check if model file exists
const MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "tests", "models", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
if !isfile(MODEL_PATH)
    println("❌ Model file not found: $MODEL_PATH")
    println("Please set INFERNO_MODEL environment variable or place model at tests/models/")
    exit(1)
end

println("✓ Model file found: $MODEL_PATH")
println("  Size: $(filesize(MODEL_PATH) ÷ 1024 ÷ 1024) MB")
println()

# Run the debugging tests
try
    include("tests/debug_inference.jl")
    println("\n✓ All debugging tests completed!")
catch e
    println("\n❌ Debugging tests failed:")
    println(e)
    exit(1)
end

println()
println("=== QUICK VALIDATION TEST ===")
println("Running basic inference test...")

try
    # Quick test like the one in the memory
    result = read(`echo "What is 2+2?" | julia --project=. examples/simple_prompt.jl`, String)
    println("✓ Basic inference test completed")
    println("Output:")
    println(result)
catch e
    println("❌ Basic inference test failed: $e")
end

println()
println("=== DEBUGGING COMPLETE ===")
println("Check the output above for any failures or warnings.")
