"""
Safetensors loader for Inferno.jl
Loads model weights from HuggingFace safetensors format.
"""
module Safetensors

using JSON3
using LinearAlgebra
using ..ModelCPU
using ..Tokenizer

export load_safetensors_model, SafetensorsFile

struct SafetensorsFile
    path::String
    metadata::Dict{String, Any}
    tensors::Dict{String, Tuple{Int, Int, Vector{Int}}}  # name -> (offset, dtype, shape)
    data::Vector{UInt8}
end

"""
Parse safetensors file header and return tensor info.
"""
function parse_safetensors(path::String)
    data = read(path)
    
    # First 8 bytes are header size (little-endian uint64)
    header_size = reinterpret(UInt64, data[1:8])[1]
    
    # Parse JSON header
    header_bytes = data[9:8+header_size]
    header = JSON3.read(String(header_bytes))
    
    # Extract tensor info
    tensors = Dict{String, Tuple{Int, Int, Vector{Int}}}()
    data_offset = 8 + header_size
    
    for (name_sym, info) in header
        name = String(name_sym)  # Convert Symbol to String
        if name == "__metadata__"
            continue
        end
        
        # Get tensor properties
        dtype = get(info, :data_type, 1)  # 1 = F32, 2 = F16
        shape = Int.(get(info, :shape, []))
        offsets = Int.(get(info, :data_offsets, [0, 0]))
        
        # Store offset relative to data section (add 1 for Julia 1-indexing)
        # The offsets in JSON are 0-indexed relative to data section start
        tensors[name] = (data_offset + offsets[1] + 1, dtype, shape)
    end
    
    metadata = get(header, "__metadata__", Dict{String, Any}())
    
    return SafetensorsFile(path, metadata, tensors, data)
end

"""
Get tensor data from safetensors file.
"""
function get_tensor(safetensors::SafetensorsFile, name::String)
    if !haskey(safetensors.tensors, name)
        return nothing
    end
    
    offset, dtype, shape = safetensors.tensors[name]
    num_elements = prod(shape)
    
    if dtype == 1  # F32
        bytes = safetensors.data[offset:offset + num_elements * 4 - 1]
        data = reinterpret(Float32, bytes)
        # Materialize the array before reshape
        return copy(reshape(Vector{Float32}(data), shape...))
    elseif dtype == 2  # F16
        bytes = safetensors.data[offset:offset + num_elements * 2 - 1]
        data16 = reinterpret(Float16, bytes)
        data = Float32.(Vector{Float16}(data16))
        return reshape(data, shape...)
    else
        error("Unsupported dtype: $dtype")
    end
end

"""
List all tensor names matching a pattern.
"""
function list_tensors(safetensors::SafetensorsFile, pattern::Regex)
    return filter(name -> occursin(pattern, name), keys(safetensors.tensors))
end

"""
Load Qwen3.5 model from safetensors format.
Accepts either a directory path (containing config.json and .safetensors files)
or a direct path to a .safetensors file.
"""
function load_safetensors_model(model_path::String)
 # Determine if path is file or directory
 if isfile(model_path) && endswith(lowercase(model_path), ".safetensors")
 # Direct file path
 safetensors_path = model_path
 model_dir = dirname(model_path)
 if isempty(model_dir)
 model_dir = "."
 end
 elseif isdir(model_path)
 # Directory path
 model_dir = model_path
 safetensors_files = filter(f -> endswith(lowercase(f), ".safetensors"), readdir(model_dir))
 if isempty(safetensors_files)
 error("No safetensors files found in $model_dir")
 end
 safetensors_path = joinpath(model_dir, safetensors_files[1])
 else
 error("Invalid path: $model_path (must be .safetensors file or directory)")
 end
 
 # Load config
 config_path = joinpath(model_dir, "config.json")
 if !isfile(config_path)
 error("config.json not found in $model_dir")
 end
 config = JSON3.read(read(config_path, String))
    
    # Parse text_config
    text_config = get(config, "text_config", Dict{String, Any}())
    
    # Create QwenConfigCPU
    model_config = ModelCPU.QwenConfigCPU(
        architecture = :qwen35,
        vocab_size = get(text_config, "vocab_size", 248320),
        hidden_size = get(text_config, "hidden_size", 1024),
        intermediate_size = get(text_config, "intermediate_size", 3584),
        num_hidden_layers = get(text_config, "num_hidden_layers", 24),
        num_attention_heads = get(text_config, "num_attention_heads", 8),
        num_key_value_heads = get(text_config, "num_key_value_heads", 2),
        head_dim = get(text_config, "head_dim", 256),
        rms_norm_eps = Float32(get(text_config, "rms_norm_eps", 1e-6)),
        rope_theta = Float32(get(text_config, "rope_theta", 10000000.0)),
        max_position_embeddings = get(text_config, "max_position_embeddings", 262144),
        partial_rotary_factor = Float32(get(get(text_config, "rope_parameters", Dict{String, Any}()), "partial_rotary_factor", 0.25)),
        full_attention_interval = get(text_config, "full_attention_interval", 4),
        ssm_inner_size = get(text_config, "intermediate_size", 3584),
        ssm_state_size = get(text_config, "linear_key_head_dim", 128),
 ssm_group_count = get(text_config, "linear_num_key_heads", 16),
 ssm_time_step_rank = get(text_config, "linear_num_value_heads", 16),
 ssm_conv_kernel = get(text_config, "linear_conv_kernel_dim", 4),
 )
 
 # Load safetensors file
 println("Loading safetensors from: $safetensors_path")
 sf = parse_safetensors(safetensors_path)
    
    # Print tensor names for debugging
    println("\nAvailable tensor names (first 20):")
    for (i, name) in enumerate(keys(sf.tensors))
        if i > 20
            break
        end
        println("  $name")
    end
    
    # Load embedding
    embed = get_tensor(sf, "model.language_model.embed_tokens.weight")
    if embed === nothing
        error("Could not find embed_tokens.weight")
    end
    embed = Matrix(Float32.(embed'))
    
    println("\nEmbedding shape: ", size(embed))
    
    # Load layers
    layers = ModelCPU.DecoderLayerCPU[]
    layer_types = get(text_config, "layer_types", [])
    
    for layer_idx in 0:(model_config.num_hidden_layers - 1)
        println("Loading layer $layer_idx...")
        
        # Determine layer type
        is_ssm = layer_idx < length(layer_types) && layer_types[layer_idx + 1] == "linear_attention"
        
 # Load attention norm - try multiple naming conventions
 # Filter out mtp layers - we want model.language_model.layers.X
 in_norm_w = nothing
 for name in keys(sf.tensors)
 if occursin("layers.$layer_idx", name) && !occursin("mtp", name)
 if occursin("input_layernorm", name) || occursin("attention_norm", name)
 in_norm_w = get_tensor(sf, name)
 break
 end
 end
 end
if in_norm_w === nothing
 in_norm_w = ones(Float32, model_config.hidden_size)
else
 # HuggingFace stores RMSNorm weights as (w-1), need to add 1
 # (llama.cpp converter does this: data_torch = data_torch + 1)
 in_norm_w = in_norm_w .+ 1.0f0
end
in_norm = ModelCPU.RMSNormCPU(vec(Float32.(in_norm_w)), model_config.rms_norm_eps)
 
 # Load post attention norm
 post_norm_w = nothing
 for name in keys(sf.tensors)
 if occursin("layers.$layer_idx", name) && !occursin("mtp", name)
 if occursin("post_attention", name) || occursin("post_attention_norm", name)
 post_norm_w = get_tensor(sf, name)
 break
 end
 end
 end
if post_norm_w === nothing
 post_norm_w = ones(Float32, model_config.hidden_size)
else
 # HuggingFace stores RMSNorm weights as (w-1), need to add 1
 post_norm_w = post_norm_w .+ 1.0f0
end
post_norm = ModelCPU.RMSNormCPU(vec(Float32.(post_norm_w)), model_config.rms_norm_eps)
        
        if is_ssm
            # Load SSM/linear attention layer
            op, mlp = load_ssm_layer_safetensors(sf, layer_idx, model_config)
        else
            # Load full attention layer
            op = load_attention_layer_safetensors(sf, layer_idx, model_config)
            mlp = load_mlp_safetensors(sf, layer_idx, model_config)
        end
        
        push!(layers, ModelCPU.DecoderLayerCPU(in_norm, op, post_norm, mlp, is_ssm))
    end
    
    # Load final norm
    final_norm_w = nothing
    for name in keys(sf.tensors)
        if occursin("language_model.norm", name) || occursin("ln_f", name)
            final_norm_w = get_tensor(sf, name)
            break
        end
    end
if final_norm_w === nothing
 final_norm_w = ones(Float32, model_config.hidden_size)
else
 # HuggingFace stores RMSNorm weights as (w-1), need to add 1
 final_norm_w = final_norm_w .+ 1.0f0
end
final_norm = ModelCPU.RMSNormCPU(vec(Float32.(final_norm_w)), model_config.rms_norm_eps)
    
    println("\nFinal norm weight mean: ", sum(final_norm.weight) / length(final_norm.weight))
    
    # LM head (tied with embedding for Qwen)
    lm_head = embed'
    
    # Create RoPE
    rotary_dim = round(Int, model_config.head_dim * model_config.partial_rotary_factor)
    rope = ModelCPU.RotaryEmbeddingCPU(model_config.head_dim, model_config.rope_theta, model_config.max_position_embeddings; rotary_dim=rotary_dim)
    
    # Load tokenizer
    tokenizer = load_hf_tokenizer(model_dir)
    
    return ModelCPU.QwenModelCPU(model_config, embed, lm_head, layers, final_norm, rope), tokenizer
end

"""
Load tokenizer from HuggingFace format files.
"""
function load_hf_tokenizer(model_dir::String)
    # Try to load tokenizer.json
    tokenizer_json_path = joinpath(model_dir, "tokenizer.json")
    
    if !isfile(tokenizer_json_path)
        error("tokenizer.json not found in $model_dir")
    end
    
    tokenizer_data = JSON3.read(read(tokenizer_json_path, String))
    
    # Load vocab
    vocab_path = joinpath(model_dir, "vocab.json")
    if isfile(vocab_path)
        vocab = JSON3.read(read(vocab_path, String))
    else
        vocab = Dict{String, Int}()
    end
    
    # Load merges
    merges_path = joinpath(model_dir, "merges.txt")
    merges = String[]
    if isfile(merges_path)
        merge_lines = readlines(merges_path)
        # Skip header line if present
        start_idx = startswith(merge_lines[1], "#") || !occursin(" ", merge_lines[1]) ? 2 : 1
        merges = merge_lines[start_idx:end]
    end
    
    # Build token_to_id and id_to_token
    # Use vocab if available, otherwise use tokenizer_data
    token_to_id = Dict{String, Int}()
    
    if !isempty(vocab)
        for (token_sym, id) in vocab
            token_to_id[String(token_sym)] = id + 1  # Convert to 1-indexed
        end
    elseif haskey(tokenizer_data, :model) && haskey(tokenizer_data[:model], :vocab)
        for (token_sym, id) in tokenizer_data[:model][:vocab]
            token_to_id[String(token_sym)] = id + 1
        end
    end
    
    vocab_size = length(token_to_id)
    id_to_token = Vector{String}(undef, vocab_size)
    for (token, id) in token_to_id
        id_to_token[id] = token
    end
    
    # Build merge priority
    merge_priority = Dict{Tuple{String, String}, Int}()
    for (i, m) in enumerate(merges)
        parts = split(m, ' ', limit=2)
        if length(parts) == 2
            merge_priority[(parts[1], parts[2])] = i
        end
    end
    
    # Get special tokens
    special_tokens = String[]
    if haskey(tokenizer_data, :added_tokens)
        added_tokens = tokenizer_data[:added_tokens]
        if isa(added_tokens, Dict) || isa(added_tokens, AbstractDict)
            for (id, info) in added_tokens
                if isa(info, Dict) || isa(info, AbstractDict)
                    content = get(info, :content, "")
                else
                    content = ""
                end
                if !in(content, special_tokens) && !isempty(content)
                    push!(special_tokens, content)
                end
            end
        end
    end
    
    # Load tokenizer config for EOS/BOS
    tokenizer_config_path = joinpath(model_dir, "tokenizer_config.json")
    eos_id = -1
    bos_id = -1
    
    if isfile(tokenizer_config_path)
        tc = JSON3.read(read(tokenizer_config_path, String))
        if haskey(tc, :eos_token)
            eos_tok = tc[:eos_token]
            if isa(eos_tok, Dict)
                eos_tok = String(get(eos_tok, :content, ""))
            end
            if haskey(token_to_id, eos_tok)
                eos_id = token_to_id[eos_tok]
            end
        end
        if haskey(tc, :bos_token)
            bos_tok = tc[:bos_token]
            if isa(bos_tok, Dict)
                bos_tok = String(get(bos_tok, :content, ""))
            end
            if haskey(token_to_id, bos_tok)
                bos_id = token_to_id[bos_tok]
            end
        end
    end
    
    # Default EOS for Qwen
    if eos_id == -1
        eos_id = 248045  # <|im_end|>
    end
    
    # Determine pretokenizer
    pretokenizer = "default"
    if haskey(tokenizer_data, :normalizer) || haskey(tokenizer_data, :pre_tokenizer)
        pretokenizer = "qwen2"
    end
    
    Tokenizer.BPETokenizer(
        token_to_id,
        id_to_token,
        [(split(m, ' ', limit=2)[1], split(m, ' ', limit=2)[2]) for m in merges if length(split(m, ' ', limit=2)) == 2],
        merge_priority,
        special_tokens,
        pretokenizer,
        bos_id,
        eos_id
    )
end

function load_ssm_layer_safetensors(sf::SafetensorsFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU)
    # Find SSM tensors with flexible naming
    # Safetensors format: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, out_proj, A_log, dt_bias, norm
    in_proj = nothing
    gate_proj = nothing  # in_proj_z in safetensors
    ssm_out = nothing
    ssm_conv1d = nothing
    ssm_a = nothing
    ssm_dt_bias = nothing
    ssm_norm_w = nothing
    alpha_weight = nothing  # in_proj_a in safetensors
    beta_weight = nothing   # in_proj_b in safetensors
    
 for name in keys(sf.tensors)
 if !occursin("layers.$layer_idx", name) || occursin("mtp", name)
 continue
 end
        
        if occursin("linear_attn.in_proj_qkv", name)
            tensor = get_tensor(sf, name)
            in_proj = Matrix{Float32}(tensor)  # (6144, 1024) - no transpose needed
        elseif occursin("linear_attn.in_proj_z", name)
            tensor = get_tensor(sf, name)
            gate_proj = Matrix{Float32}(tensor)  # (2048, 1024) - no transpose needed
        elseif occursin("linear_attn.in_proj_a", name)
            tensor = get_tensor(sf, name)
            # Safetensors stores as (num_v_heads, hidden_size) = (16, 1024)
            # GGUF expects (hidden_size, num_v_heads) = (1024, 16)
            # We need to transpose
            alpha_weight = Matrix(Float32.(tensor'))  # Transpose to (hidden_size, num_v_heads)
        elseif occursin("linear_attn.in_proj_b", name)
            tensor = get_tensor(sf, name)
            # Same as alpha - transpose from (num_v_heads, hidden_size) to (hidden_size, num_v_heads)
            beta_weight = Matrix(Float32.(tensor'))
        elseif occursin("linear_attn.out_proj", name)
            tensor = get_tensor(sf, name)
            ssm_out = Matrix{Float32}(tensor)  # (1024, 2048) - no transpose needed
 elseif occursin("linear_attn.conv1d", name)
 tensor = get_tensor(sf, name)
 # conv1d is [6144, 1, 4] in safetensors = (channels, 1, kernel_size)
 # Forward pass expects (C, K) = (channels, kernel_size) = (6144, 4)
 squeezed = squeeze_middle(tensor) # (6144, 4) = (channels, kernel_size)
 ssm_conv1d = Matrix{Float32}(squeezed) # Keep as (C, K) = (6144, 4)
        elseif occursin("linear_attn.A_log", name)
            data = get_tensor(sf, name)
            ssm_a = -exp.(Float32.(vec(data)))  # A_log stores log(-A), need -exp(A_log) = A
        elseif occursin("linear_attn.dt_bias", name)
            data = get_tensor(sf, name)
            ssm_dt_bias = vec(Float32.(data))
        elseif occursin("linear_attn.norm", name)
            data = get_tensor(sf, name)
            ssm_norm_w = vec(Float32.(data))
        end
    end
    
 # Get dimensions from in_proj shape
 conv_channels = size(in_proj, 1) # 6144
 d_inner = conv_channels ÷ 3 # 2048 (QKV split)
 
 # Get num_v_heads from A_log or dt_bias
 num_v_heads = ssm_a !== nothing ? length(ssm_a) : 16
 
 # Get head dimensions
 head_v_dim = d_inner ÷ num_v_heads # 2048 / 16 = 128
 head_k_dim = ssm_norm_w !== nothing ? length(ssm_norm_w) : 128
 
 # Conv kernel - ssm_conv1d is (channels, kernel_size) = (6144, 4)
 conv_kernel = size(ssm_conv1d, 2) # 4
    
    # Create state buffers
    conv_state = zeros(Float32, conv_channels, conv_kernel)
    h = zeros(Float32, head_v_dim, head_k_dim, num_v_heads)
    
    # SSM norm
    if ssm_norm_w === nothing
        ssm_norm_w = ones(Float32, head_k_dim)
    end
    ssm_norm = ModelCPU.RMSNormCPU(ssm_norm_w, config.rms_norm_eps)
    
    # dt_bias
    if ssm_dt_bias === nothing
        ssm_dt_bias = zeros(Float32, num_v_heads)
    end
    
    # alpha and beta weights
    if alpha_weight === nothing
        alpha_weight = zeros(Float32, num_v_heads, config.hidden_size)
    end
    if beta_weight === nothing
        beta_weight = zeros(Float32, num_v_heads, config.hidden_size)
    end
    
    return ModelCPU.GatedDeltaNetCPU(
        layer_idx,
        in_proj,
        gate_proj,
        ssm_out,
        ssm_conv1d,
        alpha_weight,
        beta_weight,
        ssm_a !== nothing ? ssm_a : zeros(Float32, num_v_heads),
        ssm_dt_bias,
        ssm_norm,
        num_v_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        d_inner,
        conv_channels,
        conv_kernel,
        conv_state,
        h
    ), load_mlp_safetensors(sf, layer_idx, config)
end

# Helper function to squeeze middle dimension of 3D array
function squeeze_middle(arr::Array{Float32, 3})
    s = size(arr)
    if s[2] == 1
        return reshape(arr, s[1], s[3])
    else
        return dropdims(arr, dims=2)
    end
end

function load_attention_layer_safetensors(sf::SafetensorsFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU)
    wq = nothing
    wk = nothing
    wv = nothing
    wo = nothing
    q_norm_w = nothing
    k_norm_w = nothing
    
 for name in keys(sf.tensors)
 if !occursin("layers.$layer_idx", name) || occursin("mtp", name)
 continue
 end
        
 if occursin("self_attn.q_proj", name) && occursin("weight", name)
 tensor = get_tensor(sf, name)
 # q_proj outputs [num_heads * head_dim * 2, hidden] = [2048 * 2, 1024] = [4096, 1024]
 # This is Q + gate concatenated
 wq = Matrix{Float32}(tensor)
        elseif occursin("self_attn.k_proj", name) && occursin("weight", name)
            tensor = get_tensor(sf, name)
            wk = Matrix{Float32}(tensor)  # (n_kv * head_dim, hidden)
        elseif occursin("self_attn.v_proj", name) && occursin("weight", name)
            tensor = get_tensor(sf, name)
            wv = Matrix{Float32}(tensor)  # (n_kv * head_dim, hidden)
        elseif occursin("self_attn.o_proj", name) && occursin("weight", name)
            tensor = get_tensor(sf, name)
            wo = Matrix{Float32}(tensor)  # (hidden, n_heads * head_dim)
        elseif occursin("q_norm", name)
            q_norm_w = vec(Float32.(get_tensor(sf, name)))
        elseif occursin("k_norm", name)
            k_norm_w = vec(Float32.(get_tensor(sf, name)))
        end
    end
    
    # Default norms
    if q_norm_w === nothing
        q_norm_w = ones(Float32, config.head_dim)
    end
    if k_norm_w === nothing
        k_norm_w = ones(Float32, config.head_dim)
    end
    
    q_norm = ModelCPU.RMSNormCPU(q_norm_w, config.rms_norm_eps)
    k_norm = ModelCPU.RMSNormCPU(k_norm_w, config.rms_norm_eps)
    
    if wq === nothing || wk === nothing || wv === nothing || wo === nothing
        error("Missing attention tensors for layer $layer_idx")
    end
    
    return ModelCPU.FullAttentionCPU(
        layer_idx,
        wq, wk, wv, wo,
        q_norm, k_norm,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        Float32(1.0 / sqrt(config.head_dim))
    )
end

function load_mlp_safetensors(sf::SafetensorsFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU)
    gate_proj = nothing
    up_proj = nothing
    down_proj = nothing
    
 for name in keys(sf.tensors)
 if !occursin("layers.$layer_idx", name) || occursin("mtp", name)
 continue
 end
        
        if occursin("mlp.gate_proj", name) && occursin("weight", name)
            tensor = get_tensor(sf, name)
            # Safetensors stores as (intermediate, hidden) which is correct for weight * x
            gate_proj = Matrix{Float32}(tensor)
        elseif occursin("mlp.up_proj", name) && occursin("weight", name)
            tensor = get_tensor(sf, name)
            up_proj = Matrix{Float32}(tensor)
        elseif occursin("mlp.down_proj", name) && occursin("weight", name)
            tensor = get_tensor(sf, name)
            # down_proj stores as (hidden, intermediate), which is already correct for weight * x
            down_proj = Matrix{Float32}(tensor)
        end
    end
    
    if gate_proj === nothing || up_proj === nothing || down_proj === nothing
        error("Missing MLP tensors for layer $layer_idx")
    end
    
    return ModelCPU.MLPCPU(gate_proj, up_proj, down_proj)
end

end # module
