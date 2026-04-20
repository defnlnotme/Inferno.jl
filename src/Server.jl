"""
OpenAI-compatible HTTP server for Inferno.jl.

Provides an HTTP server with streaming and non-streaming chat completions.

# Endpoints
- `POST /v1/chat/completions`: OpenAI-compatible chat completions
- `GET /v1/models`: List available models
- `GET /health`: Health check endpoint

# Authentication
- If `auth_token` is not provided, looks for `INFERNO_API_KEY` environment variable
- If neither is set, generates a random 32-character key on startup

# Example
```julia
model, tok = load_model("model.gguf")
start_server(8080; model=model, tokenizer=tok, auth_token="secret-key")
```
"""
module Server

using HTTP
using JSON3
using StructTypes
using Sockets
using ..Engine
using ..Model
using ..Tokenizer

# ─── OpenAI-compatible schemas ───────────────────────────────────────────────

"""
Message(role, content)

Represents a chat message in OpenAI API format.

# Fields
- `role::String`: Message role ("system", "user", "assistant")
- `content::String`: Message content
"""
struct Message
 role::String
 content::String
end

"""
ChatCompletionRequest

OpenAI-compatible request structure for chat completions.

# Fields
- `model::String`: Model identifier
- `messages::Vector{Message}`: Conversation history
- `max_tokens::Union{Int,Nothing}`: Maximum tokens to generate
- `temperature::Union{Float64,Nothing}`: Sampling temperature (0.0-2.0)
- `top_p::Union{Float64,Nothing}`: Nucleus sampling threshold
- `top_k::Union{Int,Nothing}`: Top-k sampling parameter
- `stream::Union{Bool,Nothing}`: Whether to stream the response
"""
struct ChatCompletionRequest
 model::String
 messages::Vector{Message}
 max_tokens::Union{Int,Nothing}
 temperature::Union{Float64,Nothing}
 top_p::Union{Float64,Nothing}
 top_k::Union{Int,Nothing}
 stream::Union{Bool,Nothing}
 enable_thinking::Union{Bool,Nothing}
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
const AUTH_TOKEN_REF = Ref{Union{String,Nothing}}(nothing)

# ─── Handlers ────────────────────────────────────────────────────────────────

function check_auth(stream::HTTP.Stream)
    token = AUTH_TOKEN_REF[]
    if isempty(token)
        return true
    end

    auth_header = HTTP.header(stream.message, "Authorization")
    if auth_header == "Bearer $token"
        return true
    end

    HTTP.setstatus(stream, 401)
    HTTP.setheader(stream, "Content-Type" => "application/json")
    write(stream, JSON3.write(Dict("error" => "Unauthorized: Invalid or missing API key")))
    return false
end

function build_prompt(messages::Vector{Message}; enable_thinking::Bool=false)
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
 # Qwen3.5 chat template: includes thinking tokens
 # Default (enable_thinking=false): empty think block, model answers directly
 # With thinking: open think block, model produces chain-of-thought
 if enable_thinking
 push!(parts, "<|im_start|>assistant\n<think>\n")
 else
 push!(parts, "<|im_start|>assistant\n<think>\n\n</think>\n\n")
 end
 return join(parts, "\n")
end

function handle_chat(stream::HTTP.Stream)
    check_auth(stream) || return

    try
        req = stream.message

        # Read body with size limit to prevent OOM / DoS (Max 4MB)
        MAX_BODY_SIZE = 4 * 1024 * 1024
        body_bytes = UInt8[]
        while !eof(stream)
            chunk = readavailable(stream)
            append!(body_bytes, chunk)
            if length(body_bytes) > MAX_BODY_SIZE
                HTTP.setstatus(stream, 413)
                HTTP.setheader(stream, "Content-Type" => "application/json")
                HTTP.setheader(stream, "Connection" => "close")
                write(stream, JSON3.write(Dict("error" => "Payload Too Large")))
                return
            end
        end

        local body
        try
            body = JSON3.read(body_bytes, ChatCompletionRequest)
        catch e
            HTTP.setstatus(stream, 400)
            HTTP.setheader(stream, "Content-Type" => "application/json")
            write(stream, JSON3.write(Dict("error" => "Invalid JSON payload: " * sprint(showerror, e))))
            return
        end

        model = MODEL_REF[]
        tok = TOK_REF[]
        if isnothing(model) || isnothing(tok)
            HTTP.setstatus(stream, 503)
            HTTP.setheader(stream, "Content-Type" => "application/json")
            write(stream, JSON3.write(Dict("error" => "Model not loaded")))
            return
        end

 prompt = build_prompt(body.messages; enable_thinking=something(body.enable_thinking, false))
 max_tokens = something(body.max_tokens, 128)
 temperature = Float16(something(body.temperature, 0.7))
 top_p = Float16(something(body.top_p, 0.8))
 top_k = Int(something(body.top_k, 20))
        do_stream = something(body.stream, false)

        if do_stream
            handle_stream(stream, model, tok, prompt, max_tokens, temperature, top_p, top_k, body.model)
        else
            handle_completion(stream, model, tok, prompt, max_tokens, temperature, top_p, top_k, body.model)
        end
    catch e
        @error "Error in handle_chat" exception=(e, catch_backtrace())
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
    max_tokens, temperature, top_p, top_k, model_name)
    try
        prompt_ids = Tokenizer.encode(tok, prompt)
        response_text = Engine.generate(model, tok, prompt;
            max_tokens, temperature, top_p, top_k)
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
        @error "Error in handle_completion" exception=(e, catch_backtrace())
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
    max_tokens, temperature, top_p, top_k, model_name)
    HTTP.setstatus(stream, 200)
    HTTP.setheader(stream, "Content-Type" => "text/event-stream")
    HTTP.setheader(stream, "Cache-Control" => "no-cache")
    HTTP.setheader(stream, "Connection" => "keep-alive")

    # We must send headers now to start the stream
    HTTP.startwrite(stream)

    try
        id = "chatcmpl-" * string(rand(UInt32), base=16)
        token_stream = Engine.generate_stream(model, tok, prompt;
            max_tokens, temperature, top_p, top_k)

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
        @error "Error in handle_stream" exception=(e, catch_backtrace())
    end
end

# ─── Health & Models endpoints ───────────────────────────────────────────────

function handle_health(stream::HTTP.Stream)
    HTTP.setstatus(stream, 200)
    HTTP.setheader(stream, "Content-Type" => "application/json")
    write(stream, JSON3.write(Dict("status" => "ok")))
end

function handle_models(stream::HTTP.Stream)
    check_auth(stream) || return

    HTTP.setstatus(stream, 200)
    HTTP.setheader(stream, "Content-Type" => "application/json")
    write(stream, JSON3.write(Dict("data" => [Dict(
        "id" => "qwen3.5",
        "object" => "model",
        "owned_by" => "inferno"
    )])))
end

# ─── Server startup ─────────────────────────────────────────────────────────

"""
 start_server(port=8080; model, tokenizer, auth_token)

Start an OpenAI-compatible HTTP server for chat completions.

# Arguments
- `port::Int=8080`: Port to listen on
- `model`: The loaded QwenModel (GPU) or QwenModelCPU (CPU)
- `tokenizer::BPETokenizer`: The tokenizer
- `auth_token::Union{String,Nothing}=nothing`: Optional bearer token for authentication

# Endpoints
- `POST /v1/chat/completions`: Chat completions (streaming and non-streaming)
- `GET /v1/models`: List models
- `GET /health`: Health check

# Authentication
If `auth_token` is not provided:
1. Looks for `INFERNO_API_KEY` environment variable
2. If not found, generates a random 32-character key

# Example
```julia
model, tok = load_model("model.gguf")
start_server(8080; model=model, tokenizer=tok)

# With authentication
start_server(8080; model=model, tokenizer=tok, auth_token="secret-key")
```
"""
function start_server(port::Int=8080;
    model::Union{QwenModel,Nothing}=nothing,
    tokenizer::Union{BPETokenizer,Nothing}=nothing,
    auth_token::Union{String,Nothing}=nothing)
    MODEL_REF[] = model
    TOK_REF[] = tokenizer

    if !isnothing(auth_token)
        AUTH_TOKEN_REF[] = auth_token
    elseif haskey(ENV, "INFERNO_API_KEY")
        AUTH_TOKEN_REF[] = ENV["INFERNO_API_KEY"]
    else
        AUTH_TOKEN_REF[] = bytes2hex(rand(UInt8, 16))
    end

    router = HTTP.Router()
    # Register handlers as Stream handlers
    HTTP.register!(router, "POST", "/v1/chat/completions", handle_chat)
    HTTP.register!(router, "GET", "/health", handle_health)
    HTTP.register!(router, "GET", "/v1/models", handle_models)

    @info "Inferno server started" url="http://127.0.0.1:$(port)" endpoints=["POST /v1/chat/completions", "GET /v1/models", "GET /health"]

    # Use the stream-oriented serve
    HTTP.serve(router, Sockets.localhost, port; stream=true)
end

export start_server

end # module
