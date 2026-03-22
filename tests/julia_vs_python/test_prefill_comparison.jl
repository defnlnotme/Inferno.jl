"""
Test comparing prefill_prompt_tokens output between Julia and Python (transformers).

This test ensures that the Julia implementation produces the same results as
the HuggingFace transformers library for the prefill phase.

Note: This test compares token IDs and logits. Hidden state comparison would
require instrumenting the forward pass to capture intermediate states.
"""

using Test
using JSON
using Inferno
using oneAPI

const MODEL_PATH = get(ENV, "INFERNO_MODEL", "unsloth/Qwen3.5-0.8B-GGUF")
const PYTHON_SCRIPT = joinpath(@__DIR__, "prefill_comparison.py")

# Global state for model loading
const _model_state = Ref{Any}(nothing)


"""
Run the Python script to get prefill results from transformers.
"""
function get_python_prefill(model_path::String, prompt::String;
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing,String}=nothing)
    cmd_parts = ["python3", PYTHON_SCRIPT, model_path, prompt, string(use_chat_template)]
    if system_prompt !== nothing
        push!(cmd_parts, system_prompt)
    end

    result = read(`$cmd_parts`, String)
    return JSON.parse(result)
end


"""
Initialize the Julia model state.
"""
function init_julia_model()
    if _model_state[] !== nothing
        return _model_state[]
    end

    println("Loading model: $MODEL_PATH")
    model, tokenizer = Inferno.load_model(MODEL_PATH)

    _model_state[] = (; model, tokenizer)
    return _model_state[]
end


"""
Get prefill results from Julia implementation using the high-level API.
Compares token IDs and logits with Python transformers output.
"""
function get_julia_prefill(prompt::String; use_chat_template::Bool=false)
    state = init_julia_model()
    model, tokenizer = state.model, state.tokenizer

    # Encode prompt
    token_ids = Inferno.Tokenizer.encode(tokenizer, prompt)
    @assert !isempty(token_ids) "Prompt encoded to zero tokens"

    # Initialize KV caches
    caches = [Inferno.Model.init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

    # Forward pass (prefill)
    logits = Inferno.Model.forward!(model, token_ids, 0, caches)

    # Get last token's logits
    last_logits = vec(logits[:, end])

    return Dict(
        "token_ids" => token_ids,
        "logits" => Array(last_logits)[:]
    )
end


"""
Compare two arrays with tolerance for floating point differences.
"""
function arrays_match(arr1::AbstractArray, arr2::AbstractArray; atol=1e-3, rtol=1e-2)
    if length(arr1) != length(arr2)
        return false, 0.0
    end

    max_diff = maximum(abs.(arr1 .- arr2))
    match_result = all(isapprox.(arr1, arr2, atol=atol, rtol=rtol))
    return match_result, max_diff
end


@testset "Julia vs Python: Prefill Comparison" begin

    test_prompts = [
        "Hello",
        "The capital of France is",
        "What is machine learning?",
    ]

    for prompt in test_prompts
        @testset "Prompt: '$prompt'" begin

            # Get results from both implementations
            println("\nTesting prompt: '$prompt'")

            # Julia results
            julia_result = get_julia_prefill(prompt)
            println("Julia token IDs: $(julia_result["token_ids"])")
            println("Julia logits (first 5): $(julia_result["logits"][1:5])")

            # Python results (use_chat_template=false to match raw encoding)
            python_result = get_python_prefill(MODEL_PATH, prompt; use_chat_template=false)
            println("Python token IDs: $(python_result["token_ids"])")
            println("Python logits (first 5): $(python_result["logits"][1:5])")

            # Test 1: Token IDs should match exactly
            @test julia_result["token_ids"] == parse.(Int, python_result["token_ids"])

            # Test 2: Logits should match within tolerance
            julia_logits = Float32.(julia_result["logits"])
            python_logits = Float32.(python_result["logits"])

            logits_match, max_diff = arrays_match(julia_logits, python_logits; atol=1e-1, rtol=1e-1)
            println("Logits match: $logits_match, max_diff: $max_diff")
            @test logits_match
        end
    end
end
