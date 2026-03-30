module Chat

using REPL
using ..Inferno: stream_to_stdout

export chat, chat!, start_chat, Message, build_prompt

struct Message
    role::String
    content::String
end

Message(role::Symbol, content) = Message(String(role), String(content))

function build_prompt(messages::Vector{Message})
    parts = String[]
    for msg in messages
        if msg.role == "system"
            push!(parts, "<|im_start|>system\n$(msg.content)<|im_end|>")
        elseif msg.role == "user"
            push!(parts, "<|im_start|>user\n$(msg.content)<|im_end|>")
        elseif msg.role == "assistant"
            push!(parts, "<|im_start|>assistant\n$(msg.content)<|im_end|>")
        end
    end
    push!(parts, "<|im_start|>assistant\n")
    return join(parts, "\n")
end

function chat(model, tok, messages::Vector{Message}; kwargs...)
    prompt = build_prompt(messages)
    return stream_to_stdout(model, tok, prompt; stop_token=tok.eos_id, kwargs...)
end

function chat!(model, tok; 
    system_prompt::String="You are a helpful assistant.",
    kwargs...)
    
    messages = [Message(:system, system_prompt)]
    
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║           Welcome to Inferno Chat!                ║
    ╠═══════════════════════════════════════════════════╣
    ║  Type your message and press Enter to chat.       ║
    ║  Commands:                                         ║
    ║    /clear  - Clear conversation history           ║
    ║    /system - Change system prompt                 ║
    ║    /quit   - Exit chat                             ║
    ╚═══════════════════════════════════════════════════╝
    """
    
    printstyled(banner, color=:cyan, bold=true)
    println()
    
    while true
        printstyled("You> ", color=:green, bold=true)
        line = chomp(readline(stdin))
        
        if isempty(line)
            continue
        end
        
        if line == "/quit" || line == "/exit"
            printstyled("Goodbye!\n", color=:cyan)
            break
        elseif line == "/clear"
            messages = [Message(:system, system_prompt)]
            printstyled("Conversation cleared.\n", color=:yellow)
            continue
        elseif line == "/system"
            printstyled("New system prompt: ", color=:magenta)
            new_system = chomp(readline(stdin))
            if !isempty(new_system)
                system_prompt = new_system
                messages = [Message(:system, system_prompt)]
                printstyled("System prompt updated!\n", color=:green)
            end
            continue
        elseif startswith(line, "/")
            printstyled("Unknown command: $(line)\n", color=:red)
            continue
        end
        
        push!(messages, Message(:user, line))
        
        prompt = build_prompt(messages)
        
        printstyled("\nAssistant> ", color=:blue, bold=true)
        
        response = stream_to_stdout(model, tok, prompt; stop_token=tok.eos_id, kwargs...)
        
        println()
        
        push!(messages, Message(:assistant, response))
    end
end

function start_chat(model, tok; kwargs...)
    chat!(model, tok; kwargs...)
end

end
