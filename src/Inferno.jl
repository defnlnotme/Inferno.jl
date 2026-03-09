module Inferno

include("GGUF.jl")
include("Model.jl")
include("Tokenizer.jl")
include("Loader.jl")
include("Engine.jl")
include("Server.jl")

using .GGUF
using .Model
using .Tokenizer
using .Loader
using .Engine
using .Server

export load_model, start_server

"""
    load_model(path) -> (QwenModel, BPETokenizer)

Load a GGUF model file and return the constructed model + tokenizer.
"""
function load_model(path::String)
    println("Loading GGUF file: $path")
    file = GGUF.read_gguf(path)
    println("  Metadata keys: $(length(file.metadata))")
    println("  Tensors: $(length(file.tensors))")

    # GGUF uses the architecture name as prefix for model-specific keys
    arch = get(file.metadata, "general.architecture", "llm")

    config = Model.QwenConfig(
        vocab_size         = Int(get(file.metadata, "$(arch).vocab_size",
                                    length(get(file.metadata, "tokenizer.ggml.tokens", [])))),
        hidden_size        = Int(get(file.metadata, "$(arch).embedding_length", 4096)),
        intermediate_size  = Int(get(file.metadata, "$(arch).feed_forward_length", 11008)),
        num_hidden_layers  = Int(get(file.metadata, "$(arch).block_count", 32)),
        num_attention_heads = Int(get(file.metadata, "$(arch).attention.head_count", 32)),
        num_key_value_heads = Int(get(file.metadata, "$(arch).attention.head_count_kv", 32)),
        rms_norm_eps       = Float32(get(file.metadata, "$(arch).attention.layer_norm_rms_epsilon", 1.0e-6)),
        rope_theta         = Float32(get(file.metadata, "$(arch).rope.freq_base", 1000000.0)),
        max_position_embeddings = Int(get(file.metadata, "$(arch).context_length", 32768)),
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
    main(model_path; port=8080)

Load model and start the HTTP server.
"""
function main(model_path::String; port::Int=8080)
    model, tok = load_model(model_path)
    Server.start_server(port; model=model, tokenizer=tok)
end

end # module
