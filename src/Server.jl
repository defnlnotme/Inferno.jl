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
    max_tokens::Union{Int, Nothing}
    temperature::Union{Float64, Nothing}
    top_p::Union{Float64, Nothing}
    stream::Union{Bool, Nothing}
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

const MODEL_REF = Ref{Union{QwenModel, Nothing}}(nothing)
const TOK_REF = Ref{Union{BPETokenizer, Nothing}}(nothing)

# ─── Handlers ────────────────────────────────────────────────────────────────

function build_prompt(messages::Vector{Message})
    # Simple chat template for Qwen
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

function handle_chat(req::HTTP.Request)
    body = JSON3.read(req.body, ChatCompletionRequest)

    model = MODEL_REF[]
    tok = TOK_REF[]
    if isnothing(model) || isnothing(tok)
        return HTTP.Response(503, ["Content-Type" => "application/json"],
            JSON3.write(Dict("error" => "Model not loaded")))
    end

    prompt = build_prompt(body.messages)
    max_tokens = something(body.max_tokens, 128)
    temperature = Float32(something(body.temperature, 0.7))
    top_p = Float32(something(body.top_p, 0.9))
    do_stream = something(body.stream, false)

    if do_stream
        return handle_stream(req, model, tok, prompt, max_tokens, temperature, top_p, body.model)
    end

    prompt_ids = Tokenizer.encode(tok, prompt)
    response_text = Engine.generate(model, tok, prompt;
                                    max_tokens, temperature, top_p)
    completion_ids = Tokenizer.encode(tok, response_text)

    resp = ChatCompletionResponse(
        "chatcmpl-" * string(rand(UInt32), base=16),
        "chat.completion",
        round(Int, time()),
        body.model,
        [Choice(0, Message("assistant", response_text), "stop")],
        Usage(length(prompt_ids), length(completion_ids),
              length(prompt_ids) + length(completion_ids))
    )

    return HTTP.Response(200, ["Content-Type" => "application/json"],
                         JSON3.write(resp))
end

# ─── SSE Streaming ───────────────────────────────────────────────────────────

function handle_stream(req::HTTP.Request, model, tok, prompt,
                       max_tokens, temperature, top_p, model_name)
    return HTTP.Response(200,
        ["Content-Type" => "text/event-stream",
         "Cache-Control" => "no-cache",
         "Connection" => "keep-alive"]) do io
        
        input_ids = Tokenizer.encode(tok, prompt)
        if isempty(input_ids) || input_ids[1] != tok.bos_id
            pushfirst!(input_ids, tok.bos_id)
        end

        caches = make_kv_cache(model.config)
        pos = 0
        id = "chatcmpl-" * string(rand(UInt32), base=16)

        # Prefill
        positions = collect(pos:(pos + length(input_ids) - 1))
        logits = forward!(model, input_ids, positions, caches)
        pos += length(input_ids)
        last_logits = Array(logits[:, end])
        next_token = Engine.sample_token(last_logits; temperature, top_p)

        for step in 1:max_tokens
            if next_token == tok.eos_id
                break
            end

            token_str = Tokenizer.decode(tok, [next_token])
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
            write(io, "data: $(JSON3.write(chunk))\n\n")
            flush(io)

            # Next token
            logits = forward!(model, [next_token], [pos], caches)
            pos += 1
            last_logits = Array(logits[:, 1])
            next_token = Engine.sample_token(last_logits; temperature, top_p)
        end

        # Send [DONE]
        done_chunk = Dict(
            "id" => id,
            "object" => "chat.completion.chunk",
            "created" => round(Int, time()),
            "model" => model_name,
            "choices" => [Dict(
                "index" => 0,
                "delta" => Dict(),
                "finish_reason" => "stop"
            )]
        )
        write(io, "data: $(JSON3.write(done_chunk))\n\n")
        write(io, "data: [DONE]\n\n")
        flush(io)
    end
end

# ─── Health & Models endpoints ───────────────────────────────────────────────

function handle_health(req::HTTP.Request)
    HTTP.Response(200, ["Content-Type" => "application/json"],
                  JSON3.write(Dict("status" => "ok")))
end

function handle_models(req::HTTP.Request)
    HTTP.Response(200, ["Content-Type" => "application/json"],
        JSON3.write(Dict("data" => [Dict(
            "id" => "qwen3.5",
            "object" => "model",
            "owned_by" => "inferno"
        )])))
end

# ─── Server startup ─────────────────────────────────────────────────────────

function start_server(port::Int=8080;
                      model::Union{QwenModel, Nothing}=nothing,
                      tokenizer::Union{BPETokenizer, Nothing}=nothing)
    MODEL_REF[] = model
    TOK_REF[] = tokenizer

    router = HTTP.Router()
    HTTP.register!(router, "POST", "/v1/chat/completions", handle_chat)
    HTTP.register!(router, "GET", "/health", handle_health)
    HTTP.register!(router, "GET", "/v1/models", handle_models)

    println("🔥 Inferno server listening on http://127.0.0.1:$(port)")
    println("   POST /v1/chat/completions")
    println("   GET  /v1/models")
    println("   GET  /health")
    HTTP.serve(router, Sockets.localhost, port)
end

export start_server

end # module
