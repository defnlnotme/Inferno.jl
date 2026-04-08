using Inferno
using ArgParse

function parse_commandline()
    s = ArgParseSettings(description="Simple prompt with Inferno")
    
    @add_arg_table! s begin
        "--model", "-m"
            help = "Path to GGUF model file or HuggingFace repo ID (e.g., 'user/model-name')"
            arg_type = String
            default = get(ENV, "INFERNO_MODEL", "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
        "--device", "-d"
            help = "GPU device index to use (1-based, e.g., 2 for second GPU)"
            arg_type = Int
            default = 2
        "--mmproj"
            help = "Path to mmproj file"
            arg_type = String
    end
    
    return parse_args(s)
end

function main()
    # Allow Ctrl+C to interrupt gracefully
    # PID locking to ensure only one instance runs
    pid_file = "/tmp/simple_prompt.pid"
    if isfile(pid_file)
        try
            old_pid = parse(Int, read(pid_file, String))
            if old_pid != getpid()
                # Try to kill the old process
                run(`kill -9 $old_pid`, wait=false)
                sleep(0.5) # Give it a moment to release resources
            end
        catch
            # Ignore errors (e.g., file empty or process already dead)
        end
    end
    open(pid_file, "w") do f
        write(f, string(getpid()))
    end
    
    Base.exit_on_sigint(false)
    
    args = parse_commandline()
    
    # 1. Load the model and tokenizer
    model, tok = Inferno.load_model(args["model"]; 
                                    device=args["device"], 
                                    mmproj=args["mmproj"])

    # 2. Define your prompt
    println("-"^40)
    print("Enter prompt: ")
    prompt = read(stdin, String)

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

    is_stdout_tty = isa(stdout, Base.TTY)
    if is_stdout_tty
        print("\e[2m...\e[0m")
    end
    flush(stdout)

    # generate_stream yields one string token (decoded) at a time
    stream = Inferno.generate_stream(model, tok, prompt; max_tokens=256, temperature=0.0f0, top_p=1.0f0, top_k=1)
    first_token = true
    try
        for token in stream
            if first_token
                if is_stdout_tty
                    print("\b\b\b\e[K")
                end
                first_token = false
            end
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
