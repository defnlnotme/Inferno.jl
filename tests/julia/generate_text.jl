#!/usr/bin/env julia
# generate_text.jl — Generate text with Qwen3.5 GGUF model via Inferno.jl
# This is the Julia reference for text generation to compare against Python.
#
# Usage:
#     julia --project=. tests/julia/generate_text.jl "what is 2+2?" --max-tokens 32
#
# Output: token_ids, decoded_text, timing stats

using ArgParse
using Printf
using LinearAlgebra
using JSON

using Inferno
using .Inferno: Model, Tokenizer
using .Inferno.Loader: extract_tensor, get_bias_or_norm, extract_sorted_blocks
using .Inferno.Engine: generate, generate_stream

const ROOT = normpath(joinpath(@__DIR__, "../.."))
const GGUF_PATH = joinpath(ROOT, "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "prompt"
            help = "Prompt text"
            required = true
            arg_type = String
        "--max-tokens", "-n"
            help = "Max new tokens"
            arg_type = Int
            default = 32
        "--temperature", "-t"
            help = "Temperature (0=greedy)"
            arg_type = Float64
            default = 0.0
        "--top-p", "-p"
            help = "Top-p"
            arg_type = Float64
            default = 1.0
        "--top-k", "-k"
            help = "Top-k"
            arg_type = Int
            default = 1
        "--device"
            help = "GPU device (0-based), or -1 for CPU"
            arg_type = Int
            default = 0
        "--output-json"
            help = "Output as JSON"
            action = :store_true
    end
    return ArgParse.parse_args(ARGS, s)
end

function main()
    args = parse_args()
    
    prompt = args["prompt"]
    max_tokens = args["max-tokens"]
    temperature = Float32(args["temperature"])
    top_p = Float32(args["top-p"])
    top_k = args["top-k"]
    device = args["device"]
    
    println("--- Model Info ---")
    println("Model path: $GGUF_PATH")
    println("Device: $(device < 0 ? "CPU" : "GPU $device")")
    println()
    println("--- Generation Parameters ---")
    println("Prompt: $(repr(prompt))")
    println("Max tokens: $max_tokens")
    println("Temperature: $temperature")
    println("Top-p: $top_p, Top-k: $top_k")
    println()
    
    println("Loading model...")
    load_start = time()
    
    if device < 0
        model, tok = Inferno.load_model(GGUF_PATH; device=-1)
    else
        model, tok = Inferno.load_model(GGUF_PATH; device=device)
    end
    
    load_elapsed = time() - load_start
    println("Model loaded in $(round(load_elapsed; digits=2))s")
    vocab_size = length(tok.id_to_token)
    println("Vocab size: $vocab_size")
    println("BOS: $(tok.bos_id), EOS: $(tok.eos_id)")
    println()
    
    println("--- Generating ---")
    gen_start = time()
    
    generated_ids = Int[]
    generated_text = ""
    gen_elapsed = 0.0
    kv_caches = nothing

    if temperature == 0 && top_p == 1.0 && top_k == 1
        tokens = Tokenizer.encode(tok, prompt)
        tokens_int = Vector{Int}(tokens)
        println("Using greedy (temperature=0, top_k=1)")
        
        # Use the generate_stream approach
        stream = generate_stream(model, tok, prompt;
            max_tokens=max_tokens,
            temperature=Float16(0.0),
            top_p=Float16(1.0),
            top_k=1)
        
        output_parts = String[]
        for token_str in stream
            push!(output_parts, token_str)
            print(token_str)
            flush(stdout)
        end
        gen_elapsed = time() - gen_start
        generated_text = join(output_parts)
        generated_ids = Tokenizer.encode(tok, generated_text)
        println()
    else
        stream = generate_stream(model, tok, prompt;
            max_tokens=max_tokens,
            temperature=Float16(temperature),
            top_p=Float16(top_p),
            top_k=top_k)
        
        output_parts = String[]
        for token_str in stream
            push!(output_parts, token_str)
            print(token_str)
            flush(stdout)
        end
        gen_elapsed = time() - gen_start
        generated_text = join(output_parts)
        generated_ids = Tokenizer.encode(tok, generated_text)
        println()
    end
    
    println()
    println("--- Results ---")
    println("Tokens generated: $(length(generated_ids))")
    println("Elapsed: $(@sprintf("%.3f", gen_elapsed))s")
    if gen_elapsed > 0
        println("Throughput: $(@sprintf("%.1f", length(generated_ids) / gen_elapsed)) tok/s")
    end
    
    if args["output-json"]
        println(JSON.json(Dict(
            "prompt" => prompt,
            "text" => generated_text,
            "token_ids" => generated_ids,
            "tokens_generated" => length(generated_ids),
            "elapsed" => gen_elapsed,
            "tok_per_sec" => length(generated_ids) / gen_elapsed,
        )))
    end
    
    try
        Model.free_all_kv_caches!(kv_caches)
    catch
    end
    GC.gc()
end

main()
