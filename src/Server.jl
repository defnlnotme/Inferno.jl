module Server

using HTTP
using JSON3
using StructTypes
using Sockets
using ..Engine
using ..Model
using ..Tokenizer

# ─── OpenAI-compatible schemas ───────────────────────────────────────────────

struct Message
    role::String
    content::String
end

struct ChatCompletionRequest
    model::String
    messages::Vector{Message}
    max_tokens::Union{Int,Nothing}
    temperature::Union{Float64,Nothing}
    top_p::Union{Float64,Nothing}
    stream::Union{Bool,Nothing}
end

struct Choice
    index::Int
    message::Message
    finish_reason::String
end

struct Usage
    prompt_tokens::Int
    completion_tokens::Int
    total_tokens::Int
end

struct ChatCompletionResponse
    id::String
    object::String
    created::Int
    model::String
    choices::Vector{Choice}
    usage::Usage
end

StructTypes.StructType(::Type{Message}) = StructTypes.Struct()
StructTypes.StructType(::Type{ChatCompletionRequest}) = StructTypes.Struct()
StructTypes.StructType(::Type{Choice}) = StructTypes.Struct()
StructTypes.StructType(::Type{Usage}) = StructTypes.Struct()
StructTypes.StructType(::Type{ChatCompletionResponse}) = StructTypes.Struct()

# ─── Global model state (set at startup) ─────────────────────────────────────

const MODEL_REF = Ref{Union{QwenModel,Nothing}}(nothing)
const TOK_REF = Ref{Union{BPETokenizer,Nothing}}(nothing)

# ─── Handlers ────────────────────────────────────────────────────────────────

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

function handle_chat(stream::HTTP.Stream)
    try
        req = stream.message
        body_bytes = read(stream)
        body = JSON3.read(body_bytes, ChatCompletionRequest)

        model = MODEL_REF[]
        tok = TOK_REF[]
        if isnothing(model) || isnothing(tok)
            HTTP.setstatus(stream, 503)
            HTTP.setheader(stream, "Content-Type" => "application/json")
            write(stream, JSON3.write(Dict("error" => "Model not loaded")))
            return
        end

        prompt = build_prompt(body.messages)
        max_tokens = something(body.max_tokens, 128)
        temperature = Float32(something(body.temperature, 0.7))
        top_p = Float32(something(body.top_p, 0.9))
        do_stream = something(body.stream, false)

        if do_stream
            handle_stream(stream, model, tok, prompt, max_tokens, temperature, top_p, body.model)
        else
            handle_completion(stream, model, tok, prompt, max_tokens, temperature, top_p, body.model)
        end
    catch e
        @error "Error in handle_chat" exception = (e, catch_backtrace())
        try
            if !HTTP.iswritable(stream)
                HTTP.setstatus(stream, 500)
                HTTP.setheader(stream, "Content-Type" => "application/json")
            end
            write(stream, JSON3.write(Dict("error" => string(e))))
        catch
            # Stream might be closed
        end
    end
end

function handle_completion(stream::HTTP.Stream, model, tok, prompt,
    max_tokens, temperature, top_p, model_name)
    try
        prompt_ids = Tokenizer.encode(tok, prompt)
        response_text = Engine.generate(model, tok, prompt;
            max_tokens, temperature, top_p)
        completion_ids = Tokenizer.encode(tok, response_text)

        resp = ChatCompletionResponse(
            "chatcmpl-" * string(rand(UInt32), base=16),
            "chat.completion",
            round(Int, time()),
            model_name,
            [Choice(0, Message("assistant", response_text), "stop")],
            Usage(length(prompt_ids), length(completion_ids),
                length(prompt_ids) + length(completion_ids))
        )

        HTTP.setstatus(stream, 200)
        HTTP.setheader(stream, "Content-Type" => "application/json")
        write(stream, JSON3.write(resp))
    catch e
        @error "Error in handle_completion" exception = (e, catch_backtrace())
        try
            if !HTTP.iswritable(stream)
                HTTP.setstatus(stream, 500)
                HTTP.setheader(stream, "Content-Type" => "application/json")
            end
            write(stream, JSON3.write(Dict("error" => string(e))))
        catch
            # Stream might be closed
        end
    end
end

# ─── SSE Streaming ───────────────────────────────────────────────────────────

function handle_stream(stream::HTTP.Stream, model, tok, prompt,
    max_tokens, temperature, top_p, model_name)
    HTTP.setstatus(stream, 200)
    HTTP.setheader(stream, "Content-Type" => "text/event-stream")
    HTTP.setheader(stream, "Cache-Control" => "no-cache")
    HTTP.setheader(stream, "Connection" => "keep-alive")

    # We must send headers now to start the stream
    HTTP.startwrite(stream)

    try
        id = "chatcmpl-" * string(rand(UInt32), base=16)
        token_stream = Engine.generate_stream(model, tok, prompt;
            max_tokens, temperature, top_p)

        for token_str in token_stream
            chunk = Dict(
                "id" => id,
                "object" => "chat.completion.chunk",
                "created" => round(Int, time()),
                "model" => model_name,
                "choices" => [Dict(
                    "index" => 0,
                    "delta" => Dict("content" => token_str),
                    "finish_reason" => nothing
                )]
            )
            write(stream, "data: $(JSON3.write(chunk))\n\n")
        end

        # Send [DONE]
        done_chunk = Dict(
            "choices" => [Dict("index" => 0, "delta" => Dict(), "finish_reason" => "stop")]
        )
        write(stream, "data: $(JSON3.write(done_chunk))\n\n")
        write(stream, "data: [DONE]\n\n")
    catch e
        @error "Error in handle_stream" exception = (e, catch_backtrace())
    end
end

# ─── Health & Models endpoints ───────────────────────────────────────────────

function handle_health(stream::HTTP.Stream)
    HTTP.setstatus(stream, 200)
    HTTP.setheader(stream, "Content-Type" => "application/json")
    write(stream, JSON3.write(Dict("status" => "ok")))
end

function handle_models(stream::HTTP.Stream)
    HTTP.setstatus(stream, 200)
    HTTP.setheader(stream, "Content-Type" => "application/json")
    write(stream, JSON3.write(Dict("data" => [Dict(
        "id" => "qwen3.5",
        "object" => "model",
        "owned_by" => "inferno"
    )])))
end

# ─── Server startup ─────────────────────────────────────────────────────────

function start_server(port::Int=8080;
    model::Union{QwenModel,Nothing}=nothing,
    tokenizer::Union{BPETokenizer,Nothing}=nothing)
    MODEL_REF[] = model
    TOK_REF[] = tokenizer

    router = HTTP.Router()
    # Register handlers as Stream handlers
    HTTP.register!(router, "POST", "/v1/chat/completions", handle_chat)
    HTTP.register!(router, "GET", "/health", handle_health)
    HTTP.register!(router, "GET", "/v1/models", handle_models)

    println("🔥 Inferno server listening on http://127.0.0.1:$(port)")
    println("   POST /v1/chat/completions")
    println("   GET  /v1/models")
    println("   GET  /health")

    # Use the stream-oriented serve
    HTTP.serve(router, Sockets.localhost, port; stream=true)
end

export start_server

end # module
