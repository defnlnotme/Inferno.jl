using Inferno
using ArgParse

function parse_commandline()
    s = ArgParseSettings(description="Simple prompt with Inferno")
    
    @add_arg_table! s begin
        "--model", "-m"
            help = "Path to GGUF model file"
            arg_type = String
            default = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "..", "tests", "models", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        "--device", "-d"
            help = "GPU device index to use (1-based, e.g., 2 for second GPU)"
            arg_type = Int
            default = 2
    end
    
    return parse_args(s)
end

function main()
    # Allow Ctrl+C to interrupt gracefully
    Base.exit_on_sigint(false)
    
    args = parse_commandline()
    
    # 1. Load the model and tokenizer
    model, tok = Inferno.load_model(args["model"]; device=args["device"])

    # 2. Define your prompt
    println("-"^40)
    print("Enter prompt: ")
    prompt = try
        readline()
    catch e
        if e isa EOFError
            println("\nGoodbye!")
            exit(0)
        else
            rethrow(e)
        end
    end

    if isempty(prompt)
        if eof(stdin)
            println("\nGoodbye!")
            exit(0)
        end
        prompt = "The capital of France is"
    end

    # 3. Generate and print (streaming)
    println("\nGenerating response...")
    println("-"^40)
    print("Response: ")

    # generate_stream yields one string token (decoded) at a time
    stream = Inferno.generate_stream(model, tok, prompt; max_tokens=20, temperature=0.1f0)
    try
        for token in stream
            print(token)
            flush(stdout)
        end
    catch e
        if e isa InterruptException
            println("\n\nInterrupted!")
            close(stream)
        else
            rethrow(e)
        end
    end
    println("\n" * "-"^40)
end

main()
