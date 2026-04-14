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

function handle_command(input::String, messages::Vector{Pair{String, String}})
    cmd = strip(lowercase(input))
    if cmd == "/exit" || cmd == "/quit" || cmd == "/q"
        println("Goodbye!")
        return :exit
    elseif cmd == "/clear" || cmd == "/c"
        empty!(messages)
        println("🔥 Conversation history cleared.")
        return :continue
    elseif cmd == "/help" || cmd == "/h"
        println("\e[1mAvailable commands:\e[0m")
        println("  \e[33m/exit\e[0m, \e[33m/quit\e[0m, \e[33m/q\e[0m - Exit the chat")
        println("  \e[33m/clear\e[0m, \e[33m/c\e[0m      - Clear conversation history")
        println("  \e[33m/help\e[0m, \e[33m/h\e[0m       - Show this help message")
        return :continue
    end
    return :none
end

function main()
    args = parse_commandline()
    is_stdout_tty = isa(stdout, Base.TTY)
    
    device_arg = args["device"] == -1 ? nothing : args["device"]
    println("\e[1;35m🔥 Inferno Chat Interface 🔥\e[0m")

    if !ispath(args["model"])
        println("\e[31mError: Model file not found at $(args["model"])\e[0m")
        exit(1)
    end

    println("Loading model from: $(args["model"])")
    
    Base.exit_on_sigint(false)
    model, tok = try
        Inferno.load_model(args["model"]; device=device_arg)
    catch e
        if e isa InterruptException
            println("\n\e[31m[Loading Interrupted]\e[0m")
            exit(0)
        else
            rethrow(e)
        end
    end
    
    messages = Pair{String, String}[]
    
    # If initial prompt is given, process it first
    if args["prompt"] !== nothing
        push!(messages, "user" => args["prompt"])
        prompt_text = format_messages(args["system-prompt"], messages)
        
        print("\n\e[1;32mAssistant:\e[0m ")
        if is_stdout_tty
            print("\e[2m...\e[0m")
        end
        flush(stdout)
        stream = Inferno.Engine.generate_stream(model, tok, prompt_text; 
                                              max_tokens=args["max-tokens"], 
                                              temperature=Float32(args["temperature"]), 
                                              top_p=Float32(args["top-p"]))
        
        full_response = ""
        first_token = true
        try
            for token_text in stream
                if first_token && is_stdout_tty
                    print("\b\b\b\e[K")
                    flush(stdout)
                    first_token = false
                end
                print(token_text)
                full_response *= token_text
                flush(stdout)
            end
            if first_token && is_stdout_tty
                print("\b\b\b\e[K")
                flush(stdout)
            end
            println()
        catch e
            if first_token && is_stdout_tty
                print("\b\b\b\e[K")
                flush(stdout)
            end
            if e isa InterruptException
                close(stream)
                println("\n\e[31m[Interrupted]\e[0m")
                full_response *= " [Interrupted]"
            else
                rethrow(e)
            end
        end
        push!(messages, "assistant" => full_response)
    end
    
    println("\nType \e[33m/help\e[0m for commands, or just chat away!")
    
    is_tty = isa(stdin, Base.TTY)
    
    # Interactive loop
    while true
        print("\e[1;36mUser:\e[0m ")
        flush(stdout)
        
        user_input = try
            readline()
        catch e
            if e isa InterruptException
                println("^C")
                continue
            elseif e isa EOFError
                # Handled below by checking for nothing, but catch here for completeness
                break
            else
                rethrow(e)
            end
        end
        
        if user_input === nothing || (isempty(strip(user_input)) && eof(stdin))
            # Handle EOF (Ctrl+D) gracefully
            println("Goodbye!")
            break
        elseif isempty(strip(user_input))
            continue
        end

        if !is_tty
            println(user_input)
        end

        # Handle slash commands
        if startswith(strip(user_input), "/")
            cmd_res = handle_command(user_input, messages)
            if cmd_res == :exit
                break
            elseif cmd_res == :continue
                continue
            end
            # else: might be an unknown command or just a message starting with /
        end

        # Handle legacy exit commands (optional but good for transition)
        if lowercase(strip(user_input)) in ["exit", "quit", "\\q"]
            println("Use \e[33m/exit\e[0m to quit. Goodbye!")
            break
        end
        
        push!(messages, "user" => user_input)
        prompt_text = format_messages(args["system-prompt"], messages)
        
        print("\e[1;32mAssistant:\e[0m ")
        if is_stdout_tty
            print("\e[2m...\e[0m")
        end
        flush(stdout)
        
        stream = Inferno.Engine.generate_stream(model, tok, prompt_text; 
                                              max_tokens=args["max-tokens"], 
                                              temperature=Float32(args["temperature"]), 
                                              top_p=Float32(args["top-p"]))
                                              
        full_response = ""
        first_token = true
        try
            for token_text in stream
                if first_token && is_stdout_tty
                    print("\b\b\b\e[K")
                    flush(stdout)
                    first_token = false
                end
                print(token_text)
                full_response *= token_text
                flush(stdout)
            end
            if first_token && is_stdout_tty
                print("\b\b\b\e[K")
                flush(stdout)
            end
            println()
        catch e
            if first_token && is_stdout_tty
                print("\b\b\b\e[K")
                flush(stdout)
            end
            if e isa InterruptException
                close(stream)
                println("\n\e[31m[Interrupted]\e[0m")
                full_response *= " [Interrupted]"
            else
                rethrow(e)
            end
        end
        
        push!(messages, "assistant" => full_response)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
