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

# Probe a device for the ZE_RESULT_ERROR_UNKNOWN driver corruption bug.
# Per JuliaGPU/oneAPI.jl#458: Intel Arc driver permanently corrupts the L0 command
# queue across process boundaries after mixed HostBuffer/DeviceBuffer kernel ops.
# A small probe (4 bytes) passes even on corrupted devices; medium probes (256KB) detect
# devices that fail quickly. Large uploads (>1MB) may hang indefinitely on some states.
# Returns :ok, :fast_fail, or :unknown
function probe_device(dev)::Symbol
    try
        oneAPI.device!(dev)
        # Small probe to detect immediate failures
        arr = oneArray(Int32[1])
        oneAPI.unsafe_free!(arr)
        # Medium probe to detect partial failures (256KB)
        arr2 = oneArray(zeros(Float32, 64 * 1024))
        oneAPI.unsafe_free!(arr2)
        return :ok
    catch
        return :fast_fail
    end
end

function select_device!(devs, requested_device)
    if !isnothing(requested_device)
        # Use specific requested device
        target_idx = clamp(requested_device, 1, length(devs))
        oneAPI.device!(devs[target_idx])
        println("Using requested GPU $target_idx: $(oneAPI.device())")
        return target_idx
    end
    
    # Auto-select: start from second GPU, probe devices to find working one
    # Default to GPU 2 (index 2) if available, else try others
    start_idx = min(2, length(devs))
    for offset in 0:(length(devs)-1)
        i = mod1(start_idx + offset, length(devs))
        dev = devs[i]
        status = probe_device(dev)
        if status == :ok
            oneAPI.device!(dev)
            println("Using GPU $i (probed OK): $(oneAPI.device())")
            return i
        else
            println("Skipping GPU $i (probe failed: $status)")
        end
    end
    
    # Fallback to first device if all fail
    target_idx = 1
    oneAPI.device!(devs[target_idx])
    println("All devices failed probe, falling back to GPU $target_idx: $(oneAPI.device())")
    return target_idx
end

"""
    load_model(path; device=nothing) -> (QwenModel, BPETokenizer)

Load a GGUF model file and return the constructed model + tokenizer.
"""
function load_model(path::String; device::Union{Int, Nothing}=nothing)
    devs = collect(oneAPI.devices())
    if isempty(devs)
        @warn "No oneAPI devices found."
    else
        select_device!(devs, device)
    end

    println("Loading GGUF file: $path")
    file = GGUF.read_gguf(path)
    println("  Metadata keys: $(length(file.metadata))")
    println("  Tensors: $(length(file.tensors))")

    arch_str = get(file.metadata, "general.architecture", "qwen2")
    arch = Symbol(arch_str)

    config = Model.QwenConfig(
        architecture=arch,
        vocab_size=Int(get(file.metadata, "$(arch_str).vocab_size",
            length(get(file.metadata, "tokenizer.ggml.tokens", [])))),
        hidden_size=Int(get(file.metadata, "$(arch_str).embedding_length", 1024)),
        intermediate_size=Int(get(file.metadata, "$(arch_str).feed_forward_length", 3584)),
        num_hidden_layers=Int(get(file.metadata, "$(arch_str).block_count", 24)),
        num_attention_heads=Int(get(file.metadata, "$(arch_str).attention.head_count", 8)),
        num_key_value_heads=Int(get(file.metadata, "$(arch_str).attention.head_count_kv", 2)),
        head_dim=Int(get(file.metadata, "$(arch_str).attention.key_length", 256)),
        rms_norm_eps=Float32(get(file.metadata, "$(arch_str).attention.layer_norm_rms_epsilon", 1.0e-6)),
        rope_theta=Float32(get(file.metadata, "$(arch_str).rope.freq_base", 10000000.0)),
        max_position_embeddings=min(4096, Int(get(file.metadata, "$(arch_str).context_length", 32768))),
        full_attention_interval=Int(get(file.metadata, "$(arch_str).full_attention_interval", 4)),
        ssm_inner_size=Int(get(file.metadata, "$(arch_str).ssm.inner_size", 2048)),
        ssm_state_size=Int(get(file.metadata, "$(arch_str).ssm.state_size", 128)),
        ssm_group_count=Int(get(file.metadata, "$(arch_str).ssm.group_count", 16)),
        ssm_time_step_rank=Int(get(file.metadata, "$(arch_str).ssm.time_step_rank", 16)),

        # MoE
        num_experts=Int(get(file.metadata, "$(arch_str).expert_count", 0)),
        num_experts_per_tok=Int(get(file.metadata, "$(arch_str).expert_used_count", 0)),

        # MLA
        q_lora_rank=Int(get(file.metadata, "$(arch_str).attention.q_lora_rank", 0)),
        kv_lora_rank=Int(get(file.metadata, "$(arch_str).attention.kv_lora_rank", 0)),
        qk_rope_head_dim=Int(get(file.metadata, "$(arch_str).attention.qk_rope_head_dim", 0)),
        v_head_dim=Int(get(file.metadata, "$(arch_str).attention.v_head_dim", 0)),
    )
    println("  Config: hidden=$(config.hidden_size), layers=$(config.num_hidden_layers), heads=$(config.num_attention_heads)")

    println("Loading weights...")
    Model.init_gpu_tables(QuantsData.IQ2XXS_GRID, QuantsData.KSIGNS_IQ2XS, QuantsData.KMASK_IQ2XS)
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
