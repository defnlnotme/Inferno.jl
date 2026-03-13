module Inferno

using oneAPI

include("QuantsData.jl")
include("Dequant.jl")
include("GGUF.jl")
include("Model.jl")
include("Tokenizer.jl")
include("Loader.jl")
include("Engine.jl")
include("Server.jl")

using .QuantsData
using .Dequant
using .GGUF
using .Model
using .Tokenizer
using .Loader
using .Engine
using .Server

export load_model, start_server

"""
    load_model(path; device=nothing) -> (QwenModel, BPETokenizer)

Load a GGUF model file and return the constructed model + tokenizer.
"""
function load_model(path::String; device::Union{Int, Nothing}=nothing)
    devs = collect(oneAPI.devices())
    if !isempty(devs)
        # Default to GPU 2 if available per user's rule, otherwise 1.
        target_dev = isnothing(device) ? (length(devs) >= 2 ? 2 : 1) : device
        if 1 <= target_dev <= length(devs)
            oneAPI.device!(devs[target_dev])
            println("Using GPU $target_dev: $(oneAPI.device())")
        else
            @warn "Requested device index $target_dev out of bounds. Using GPU 1."
            oneAPI.device!(devs[1])
            println("Using GPU 1: $(oneAPI.device())")
        end
    else
        @warn "No oneAPI devices found."
    end

    println("Loading GGUF file: $path")
    file = GGUF.read_gguf(path)
    println("  Metadata keys: $(length(file.metadata))")
    println("  Tensors: $(length(file.tensors))")

    # GGUF uses the architecture name as prefix for model-specific keys
    arch = get(file.metadata, "general.architecture", "llm")

    config = Model.QwenConfig(
        vocab_size=Int(get(file.metadata, "$(arch).vocab_size",
            length(get(file.metadata, "tokenizer.ggml.tokens", [])))),
        hidden_size=Int(get(file.metadata, "$(arch).embedding_length", 1024)),
        intermediate_size=Int(get(file.metadata, "$(arch).feed_forward_length", 3584)),
        num_hidden_layers=Int(get(file.metadata, "$(arch).block_count", 24)),
        num_attention_heads=Int(get(file.metadata, "$(arch).attention.head_count", 8)),
        num_key_value_heads=Int(get(file.metadata, "$(arch).attention.head_count_kv", 2)),
        head_dim=Int(get(file.metadata, "$(arch).attention.key_length", 256)),
        rms_norm_eps=Float32(get(file.metadata, "$(arch).attention.layer_norm_rms_epsilon", 1.0e-6)),
        rope_theta=Float32(get(file.metadata, "$(arch).rope.freq_base", 10000000.0)),
        max_position_embeddings=min(4096, Int(get(file.metadata, "$(arch).context_length", 32768))),
        full_attention_interval=Int(get(file.metadata, "$(arch).full_attention_interval", 4)),
        ssm_inner_size=Int(get(file.metadata, "$(arch).ssm.inner_size", 2048)),
        ssm_state_size=Int(get(file.metadata, "$(arch).ssm.state_size", 128)),
        ssm_group_count=Int(get(file.metadata, "$(arch).ssm.group_count", 16)),
        ssm_time_step_rank=Int(get(file.metadata, "$(arch).ssm.time_step_rank", 16)),
    )
    println("  Config: hidden=$(config.hidden_size), layers=$(config.num_hidden_layers), heads=$(config.num_attention_heads)")

    println("Loading weights...")
    model = Loader.load_weights(file, config)
    println("Model loaded successfully.")

    println("Loading tokenizer...")
    tok = Tokenizer.load_tokenizer(file.metadata)
    println("  Vocab size: $(length(tok.id_to_token)), BOS=$(tok.bos_id), EOS=$(tok.eos_id)")

    return model, tok
end

"""
    main(model_path; port=8080, device=nothing)

Load model and start the HTTP server.
"""
function main(model_path::String; port::Int=8080, device::Union{Int, Nothing}=nothing)
    model, tok = load_model(model_path; device=device)
    Server.start_server(port; model=model, tokenizer=tok)
end

end # module
