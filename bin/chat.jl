using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using Inferno

function parse_commandline()
    s = ArgParseSettings(description="Chat with an LLM using Inferno")

    @add_arg_table! s begin
        "--model", "-m"
            help = "Path to GGUF model file"
            arg_type = String
            required = true
        "--system-prompt", "-s"
            help = "System prompt to initialize the model's behavior"
            arg_type = String
            default = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
        "--prompt", "-p"
            help = "Initial user prompt. If provided, the model will answer it before dropping into the interactive loop."
            arg_type = String
            default = nothing
        "--max-tokens"
            help = "Maximum number of tokens to generate per response"
            arg_type = Int
            default = 128
        "--temperature", "-t"
            help = "Temperature for sampling (higher = more creative/random)"
            arg_type = Float64
            default = 0.7
        "--top-p"
            help = "Top-p sampling (nucleus sampling)"
            arg_type = Float64
            default = 0.9
        "--device", "-d"
            help = "GPU device index to use. Defaults to the second GPU if available, else first."
            arg_type = Int
            default = -1
    end

    return parse_args(s)
end

function format_messages(system_prompt::String, messages::Vector{Pair{String, String}})
    # Messages is a vector of "role" => "content"
    parts = String[]
    push!(parts, "<|im_start|>system\n$system_prompt<|im_end|>")
    
    for (role, content) in messages
        push!(parts, "<|im_start|>$role\n$content<|im_end|>")
    end
    
    # Add the final assistant prompt
    push!(parts, "<|im_start|>assistant\n")
    return join(parts, "\n")
end

function main()
    args = parse_commandline()
    
    device_arg = args["device"] == -1 ? nothing : args["device"]
    println("🔥 Inferno Chat Interface 🔥")
    println("Loading model from: $(args["model"])")
    
    model, tok = Inferno.load_model(args["model"]; device=device_arg)
    
    messages = Pair{String, String}[]
    
    # If initial prompt is given, process it first
    if args["prompt"] !== nothing
        push!(messages, "user" => args["prompt"])
        prompt_text = format_messages(args["system-prompt"], messages)
        
        print("\n\e[32mAssistant:\e[0m ")
        stream = Inferno.Engine.generate_stream(model, tok, prompt_text; 
                                              max_tokens=args["max-tokens"], 
                                              temperature=Float32(args["temperature"]), 
                                              top_p=Float32(args["top-p"]))
        
        full_response = ""
        for token_text in stream
            print(token_text)
            full_response *= token_text
            flush(stdout)
        end
        println()
        push!(messages, "assistant" => full_response)
    end
    
    println("\nType 'exit', 'quit', or '\\q' to stop.")
    
    # Interactive loop
    while true
        print("\n\e[36mUser:\e[0m ")
        user_input = readline()
        
        if isempty(strip(user_input))
            continue
        end
        if lowercase(strip(user_input)) in ["exit", "quit", "\\q"]
            println("Goodbye!")
            break
        end
        
        push!(messages, "user" => user_input)
        prompt_text = format_messages(args["system-prompt"], messages)
        
        print("\e[32mAssistant:\e[0m ")
        stream = Inferno.Engine.generate_stream(model, tok, prompt_text; 
                                              max_tokens=args["max-tokens"], 
                                              temperature=Float32(args["temperature"]), 
                                              top_p=Float32(args["top-p"]))
                                              
        full_response = ""
        for token_text in stream
            print(token_text)
            full_response *= token_text
            flush(stdout)
        end
        println()
        push!(messages, "assistant" => full_response)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
