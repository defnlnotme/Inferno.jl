#!/usr/bin/env julia
# benchmark_flash_attention_e2e.jl - End-to-end model benchmark
# Usage: INFERNO_USE_FLASH_ATTENTION=false julia --project=. benchmark/flash_attention_e2e.jl
#        INFERNO_USE_FLASH_ATTENTION=true julia --project=. benchmark/flash_attention_e2e.jl

using Inferno, Printf, LinearAlgebra, Statistics, Random, Dates
Random.seed!(42)

const GGUF_PATH = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"

function run_benchmark(model_path::String; max_tokens=128)
    println("="^70)
    println("Flash Attention End-to-End Benchmark")
    println("="^70)
    println("Model: Qwen3.5-0.8B")
    println("Flash Attention: ", get(ENV, "INFERNO_USE_FLASH_ATTENTION", "true") != "false" ? "ENABLED" : "DISABLED")
    println("-"^70)
    
    println("\nLoading model from: $model_path")
    model, tok = Inferno.load_model_cpu(model_path; keep_quantized=true)
    config = model.config
    println("  $(config.num_hidden_layers) layers ($(config.num_hidden_layers - div(config.num_hidden_layers,4)) SSM + $(div(config.num_hidden_layers,4)) Attention)")
    println("  Hidden size: $(config.hidden_size), Heads: $(config.num_attention_heads)")
    println()
    
    prompts = [
        "What is 2 + 2?",
        "The capital of France is",
        "Hello, my name is",
        "In the year 1492, Christopher Columbus",
        "To understand quantum mechanics, we must first"
    ]
    
    all_times = Float64[]
    all_tokens_per_sec = Float64[]
    
    for (i, prompt) in enumerate(prompts)
        print("Test $i/$(length(prompts)): \"$prompt\" ... ")
        
        tokens = Inferno.Tokenizer.encode(tok, prompt)
        result = Inferno.generate(model, tokens; max_tokens=max_tokens, temperature=0.0)
        
        total_time = result.total_time
        generated_tokens = length(result.tokens) - length(tokens)
        tokens_per_sec = generated_tokens / total_time
        
        push!(all_times, total_time)
        push!(all_tokens_per_sec, tokens_per_sec)
        
        @printf "%.2fs, %d tokens (%.2f tok/s)\n" total_time generated_tokens tokens_per_sec
    end
    
    println()
    println("-"^70)
    println("Summary Statistics:")
    println("  Mean time: ", @sprintf("%.3f", mean(all_times)), " ± ", @sprintf("%.3f", std(all_times)), " s")
    println("  Mean tokens/sec: ", @sprintf("%.2f", mean(all_tokens_per_sec)), " ± ", @sprintf("%.2f", std(all_tokens_per_sec)))
    println("  Median tokens/sec: ", @sprintf("%.2f", median(all_tokens_per_sec)))
    println("="^70)
    
    return mean(all_tokens_per_sec)
end

# Run benchmark
tok_per_sec = run_benchmark(GGUF_PATH; max_tokens=64)

# Write result to file for comparison
fa_enabled = get(ENV, "INFERNO_USE_FLASH_ATTENTION", "true") != "false"
mode = fa_enabled ? "flash" : "standard"
open("benchmark/e2e_$(mode)_results.txt", "w") do f
    println(f, "Tokens/sec: $tok_per_sec")
    println(f, "Flash Attention: $fa_enabled")
    println(f, "Timestamp: ", Dates.now())
end
