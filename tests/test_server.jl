module Engine
    export generate, generate_stream
    function generate(model, tok, prompt; max_tokens=128, temperature=0.7, top_p=0.9)
        return "Mocked response"
    end
    function generate_stream(model, tok, prompt; max_tokens=128, temperature=0.7, top_p=0.9)
        return Channel{String}(ch -> begin
            put!(ch, "Mocked ")
            put!(ch, "stream ")
            put!(ch, "response")
            close(ch)
        end)
    end
end

module Model
    export QwenModel, init_gpu_tables
    struct QwenModel end
    function init_gpu_tables(grid, ksigns, kmask) end
end

module Tokenizer
    export BPETokenizer, encode, decode
    struct BPETokenizer end
    function encode(tok, text) return [1, 2, 3] end
    function decode(tok, ids) return "Mocked response" end
end

include("../src/Server.jl")

using Test
using Sockets
using HTTP
using JSON3
using StructTypes
using .Server

@testset "Server Endpoints" begin
    # Create router
    router = HTTP.Router()
    HTTP.register!(router, "POST", "/v1/chat/completions", Server.handle_chat)
    HTTP.register!(router, "GET", "/health", Server.handle_health)
    HTTP.register!(router, "GET", "/v1/models", Server.handle_models)

    # Start server on dynamic port
    port, server = Sockets.listenany(8085)
    server_task = @async HTTP.serve(router, "127.0.0.1", port; stream=true, server=server)
    sleep(1) # wait for server to start

    base_url = "http://127.0.0.1:$port"

    @testset "Health Endpoint" begin
        resp = HTTP.get("$base_url/health")
        @test resp.status == 200
        data = JSON3.read(resp.body)
        @test data.status == "ok"
    end

    @testset "Models Endpoint" begin
        resp = HTTP.get("$base_url/v1/models")
        @test resp.status == 200
        data = JSON3.read(resp.body)
        @test length(data.data) > 0
        @test data.data[1].id == "qwen3.5"
    end

    @testset "Chat Completions - Model Not Loaded" begin
        # Clear model refs
        Server.MODEL_REF[] = nothing
        Server.TOK_REF[] = nothing

        req_body = Dict(
            "model" => "qwen3.5",
            "messages" => [
                Dict("role" => "user", "content" => "Hello")
            ]
        )

        # Test error response when model is missing
        try
            HTTP.post("$base_url/v1/chat/completions",
                ["Content-Type" => "application/json"],
                JSON3.write(req_body))
            @test false # Should not reach here
        catch e
            @test e isa HTTP.Exceptions.StatusError
            @test e.status == 503
            err_data = JSON3.read(e.response.body)
            @test err_data.error == "Model not loaded"
        end
    end

    @testset "Chat Completions - Non-Streaming" begin
        # Set dummy model refs
        Server.MODEL_REF[] = Model.QwenModel()
        Server.TOK_REF[] = Tokenizer.BPETokenizer()

        req_body = Dict(
            "model" => "qwen3.5",
            "messages" => [
                Dict("role" => "system", "content" => "You are a helpful assistant."),
                Dict("role" => "user", "content" => "Hello")
            ],
            "max_tokens" => 50,
            "temperature" => 0.8,
            "top_p" => 0.95,
            "stream" => false
        )

        resp = HTTP.post("$base_url/v1/chat/completions",
            ["Content-Type" => "application/json"],
            JSON3.write(req_body))

        @test resp.status == 200
        data = JSON3.read(resp.body)

        @test data.object == "chat.completion"
        @test data.model == "qwen3.5"
        @test length(data.choices) == 1
        @test data.choices[1].message.role == "assistant"
        @test data.choices[1].message.content == "Mocked response"
        @test data.choices[1].finish_reason == "stop"

        # Check usage tokens
        @test data.usage.prompt_tokens == 3 # based on our mock encode length
        @test data.usage.completion_tokens == 3 # based on our mock encode length
        @test data.usage.total_tokens == 6
    end

    @testset "Chat Completions - Streaming" begin
        Server.MODEL_REF[] = Model.QwenModel()
        Server.TOK_REF[] = Tokenizer.BPETokenizer()

        req_body = Dict(
            "model" => "qwen3.5",
            "messages" => [
                Dict("role" => "user", "content" => "Tell me a story.")
            ],
            "stream" => true
        )

        # HTTP stream request
        resp = HTTP.post("$base_url/v1/chat/completions",
            ["Content-Type" => "application/json"],
            JSON3.write(req_body))

        body_str = String(resp.body)

        chunks = String[]
        for line in split(body_str, "\n")
            if startswith(line, "data: ")
                push!(chunks, line[7:end])
            end
        end

        @test length(chunks) == 5 # 3 stream chunks + 1 empty finish_reason chunk + 1 [DONE]

        # Verify middle chunks
        chunk1 = JSON3.read(chunks[1])
        @test chunk1.choices[1].delta.content == "Mocked "
        @test chunk1.object == "chat.completion.chunk"

        chunk2 = JSON3.read(chunks[2])
        @test chunk2.choices[1].delta.content == "stream "

        chunk3 = JSON3.read(chunks[3])
        @test chunk3.choices[1].delta.content == "response"

        chunk4 = JSON3.read(chunks[4])
        @test chunk4.choices[1].finish_reason == "stop"

        @test chunks[5] == "[DONE]"
    end

    # Cleanup
    close(server)
end
