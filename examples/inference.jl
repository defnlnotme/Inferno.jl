if isdefined(@__MODULE__, :TestInference)
    old_mod = getfield(@__MODULE__, :TestInference)
    try
        if isdefined(old_mod, :free_all_gpu!)
            Base.invokelatest(getfield(old_mod, :free_all_gpu!))
        end
    catch err
        @warn "Failed to free GPU state from previous TestInference module during reload" exception=(err, catch_backtrace())
    end
    try
        if isdefined(old_mod, :oneAPI)
            getfield(old_mod, :oneAPI).synchronize()
        end
    catch
    end
    try
        GC.gc(true)
    catch
    end
end

module TestInference
using oneAPI
using LinearAlgebra
using Inferno
using .Inferno: Model
 using .Inferno.Loader: get_bias_or_norm, extract_tensor, get_weight, BlockTensors, extract_sorted_blocks
 include("cleanup_memory.jl")
 include("inference_trace.jl")



# step 0 prompt and model
prompt = "what is 2 + 2?"
MODEL_PATH = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
const file = Inferno.GGUF.read_gguf(MODEL_PATH)
const tokenizer = Inferno.Tokenizer.load_tokenizer(file.metadata)
const chat_template = let
    for key in ("tokenizer.chat_template", "tokenizer.ggml.chat_template", "chat_template")
        if haskey(file.metadata, key)
            return String(file.metadata[key])
        end
    end
    nothing
end

function get_chat_template()
    if isdefined(@__MODULE__, :chat_template)
        return getfield(@__MODULE__, :chat_template)
    end
    for key in ("tokenizer.chat_template", "tokenizer.ggml.chat_template", "chat_template")
        if haskey(file.metadata, key)
            return String(file.metadata[key])
        end
    end
    return nothing
end

# step 1 embedding vector
const embed = extract_tensor(file, "token_embd.weight")
const embed_gpu = get_weight(file, "token_embd.weight")

arch_str = get(file.metadata, "general.architecture", "qwen35")
arch = Symbol(arch_str)

function require_metadata(file, key)
    haskey(file.metadata, key) || error("Missing required GGUF metadata key: $key")
    return file.metadata[key]
end

function build_config(file)
    return Model.QwenConfig(
        architecture=arch,
        vocab_size=Int(get(file.metadata, "$(arch_str).vocab_size", length(get(file.metadata, "tokenizer.ggml.tokens", [])))),
        hidden_size=Int(require_metadata(file, "$(arch_str).embedding_length")),
        intermediate_size=Int(require_metadata(file, "$(arch_str).feed_forward_length")),
        num_hidden_layers=Int(require_metadata(file, "$(arch_str).block_count")),
        num_attention_heads=Int(require_metadata(file, "$(arch_str).attention.head_count")),
        num_key_value_heads=Int(require_metadata(file, "$(arch_str).attention.head_count_kv")),
        head_dim=Int(require_metadata(file, "$(arch_str).attention.key_length")),
        rms_norm_eps=Float16(require_metadata(file, "$(arch_str).attention.layer_norm_rms_epsilon")),
        rope_theta=Float32(require_metadata(file, "$(arch_str).rope.freq_base")),
        max_position_embeddings=min(4096, Int(require_metadata(file, "$(arch_str).context_length"))),
        full_attention_interval=Int(require_metadata(file, "$(arch_str).full_attention_interval")),
        ssm_inner_size=Int(require_metadata(file, "$(arch_str).ssm.inner_size")),
        ssm_state_size=Int(require_metadata(file, "$(arch_str).ssm.state_size")),
        ssm_group_count=Int(require_metadata(file, "$(arch_str).ssm.group_count")),
        ssm_time_step_rank=Int(require_metadata(file, "$(arch_str).ssm.time_step_rank")),
        ssm_conv_kernel=Int(require_metadata(file, "$(arch_str).ssm.conv_kernel")),

        # MoE
        num_experts=Int(get(file.metadata, "$(arch_str).expert_count", 0)),
        num_experts_per_tok=Int(get(file.metadata, "$(arch_str).expert_used_count", 0)),

        # MLA
        q_lora_rank=Int(get(file.metadata, "$(arch_str).attention.q_lora_rank", 0)),
        kv_lora_rank=Int(get(file.metadata, "$(arch_str).attention.kv_lora_rank", 0)),
        qk_rope_head_dim=Int(get(file.metadata, "$(arch_str).attention.qk_rope_head_dim", 0)),
        v_head_dim=Int(get(file.metadata, "$(arch_str).attention.v_head_dim", 0)),
    )
end

runtime_config() = isdefined(@__MODULE__, :config) ? getfield(@__MODULE__, :config) : build_config(file)

const config = build_config(file)

function validate_config(config)
    config.num_attention_heads > 0 || error("num_attention_heads must be positive")
    config.num_key_value_heads > 0 || error("num_key_value_heads must be positive")
    config.ssm_group_count > 0 || error("ssm_group_count must be positive")
    config.ssm_time_step_rank > 0 || error("ssm_time_step_rank must be positive")
    config.head_dim > 0 || error("head_dim must be positive")
    config.num_attention_heads % config.num_key_value_heads == 0 ||
        error("Grouped-query attention requires num_attention_heads divisible by num_key_value_heads")
    config.ssm_inner_size % config.ssm_time_step_rank == 0 ||
        error("SSM inner size must be divisible by ssm_time_step_rank")
    conv_channels = 2 * config.ssm_group_count * config.ssm_state_size + config.ssm_inner_size
    conv_channels > 0 || error("SSM conv channel count must be positive")
    return nothing
end

Base.invokelatest(validate_config, runtime_config())

# Pre-loaded weights structure to avoid allocations during forward pass
struct LoadedBlock
    index::Int
    is_ssm::Bool

    # Normalization
    attn_norm::Model.RMSNorm
    post_attn_norm::Model.RMSNorm

    # SSM Weights (if is_ssm)
    attn_qkv_weight::Union{AbstractMatrix, Nothing}
    attn_gate_weight::Union{AbstractMatrix, Nothing}
    ssm_out_weight::Union{AbstractMatrix, Nothing}
    ssm_conv1d_weight::Union{AbstractMatrix, Nothing}
    ssm_alpha_weight::Union{AbstractMatrix, Nothing}
    ssm_beta_weight::Union{AbstractMatrix, Nothing}
    ssm_a::Union{AbstractVector, Nothing}
    ssm_dt_bias::Union{AbstractVector, Nothing}
    ssm_alpha_weight_cpu::Union{Matrix{Float32}, Nothing}
    ssm_beta_weight_cpu::Union{Matrix{Float32}, Nothing}
    ssm_a_cpu::Union{Vector{Float32}, Nothing}
    ssm_dt_bias_cpu::Union{Vector{Float32}, Nothing}
    ssm_norm::Union{Model.RMSNorm, Nothing}

    # Attention Weights (if !is_ssm)
    attn_q_weight::Union{AbstractMatrix, Nothing}
    attn_k_weight::Union{AbstractMatrix, Nothing}
    attn_v_weight::Union{AbstractMatrix, Nothing}
    attn_output_weight::Union{AbstractMatrix, Nothing}
    attn_q_norm::Union{Model.RMSNorm, Nothing}
    attn_k_norm::Union{Model.RMSNorm, Nothing}

    # MLP Weights
    ffn_gate_weight::AbstractMatrix
    ffn_up_weight::AbstractMatrix
    ffn_down_weight::AbstractMatrix
end

function load_all_blocks(file, blocks, config)
    loaded = LoadedBlock[]
    for b in blocks
        is_ssm = b.ssm_a !== nothing

        # Norms
        attn_norm = Model.RMSNorm(get_bias_or_norm(file, b.attn_norm_weight), config.rms_norm_eps)
        post_norm_info = b.post_attention_norm_weight !== nothing ? b.post_attention_norm_weight :
                         b.ffn_norm_weight !== nothing ? b.ffn_norm_weight :
                         b.post_attention_layernorm_weight !== nothing ? b.post_attention_layernorm_weight :
                         b.attn_norm_weight
        post_attn_norm = Model.RMSNorm(get_bias_or_norm(file, post_norm_info), config.rms_norm_eps)

        # MLP
        ffn_gate = get_weight(file, b.ffn_gate_weight)
        ffn_up = get_weight(file, b.ffn_up_weight)
        ffn_down = get_weight(file, b.ffn_down_weight)

        if is_ssm
            push!(loaded, LoadedBlock(
                b.index, true, attn_norm, post_attn_norm,
                get_weight(file, b.attn_qkv_weight),
                get_weight(file, b.attn_gate_weight),
                get_weight(file, b.ssm_out_weight),
                oneArray(Inferno.Loader.extract_tensor(file, b.ssm_conv1d_weight)'), # Pre-transposed
                get_weight(file, b.ssm_alpha_weight),
                get_weight(file, b.ssm_beta_weight),
                get_bias_or_norm(file, b.ssm_a),
                get_bias_or_norm(file, b.ssm_dt_bias),
                Matrix{Float32}(Array(get_weight(file, b.ssm_alpha_weight))),
                Matrix{Float32}(Array(get_weight(file, b.ssm_beta_weight))),
                Vector{Float32}(Array(get_bias_or_norm(file, b.ssm_a))),
                Vector{Float32}(Array(get_bias_or_norm(file, b.ssm_dt_bias))),
                Model.RMSNorm(get_bias_or_norm(file, b.ssm_norm_weight), config.rms_norm_eps),
                nothing, nothing, nothing, nothing, nothing, nothing,
                ffn_gate, ffn_up, ffn_down
            ))
        else
            push!(loaded, LoadedBlock(
                b.index, false, attn_norm, post_attn_norm,
                nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                nothing, nothing, nothing, nothing, nothing,
                get_weight(file, b.attn_q_weight),
                get_weight(file, b.attn_k_weight),
                get_weight(file, b.attn_v_weight),
                get_weight(file, b.attn_output_weight),
                Model.RMSNorm(get_bias_or_norm(file, b.attn_q_norm_weight), config.rms_norm_eps),
                Model.RMSNorm(get_bias_or_norm(file, b.attn_k_norm_weight), config.rms_norm_eps),
                ffn_gate, ffn_up, ffn_down
            ))
        end
    end
    return loaded
end

const blocks_meta = extract_sorted_blocks(file.tensors)
const loaded_blocks = load_all_blocks(file, blocks_meta, runtime_config())
const lm_head_weight = haskey(file.tensors, "output.weight") ?
                     get_weight(file, "output.weight") :
                     get_weight(file, "token_embd.weight")
const logits_work_buf = oneArray{Float16}(undef, size(lm_head_weight, 1))
const logits_host = Vector{Float16}(undef, size(lm_head_weight, 1))
const final_norm_w_loaded = get_bias_or_norm(file, "output_norm.weight")
const final_rms_pre = let cfg = runtime_config()
    Model.RMSNorm(final_norm_w_loaded, cfg.rms_norm_eps)
end
const rope_dim = let cfg = runtime_config()
    cfg.qk_rope_head_dim > 0 ? cfg.qk_rope_head_dim : cfg.head_dim
end
const rope_pairs = rope_dim ÷ 2
const rope_sin_cache = let
    cfg = runtime_config()
    cache = Matrix{Float32}(undef, rope_pairs, cfg.max_position_embeddings)
    theta = Float32(cfg.rope_theta)
    for pair_idx in 1:rope_pairs
        i = 2 * pair_idx - 1
        freq = Float32(1.0) / (theta ^ (Float32(i - 1) / Float32(rope_dim)))
        for pos in 1:cfg.max_position_embeddings
            cache[pair_idx, pos] = sin(Float32(pos - 1) * freq)
        end
    end
    cache
end
const rope_cos_cache = let
    cfg = runtime_config()
    cache = Matrix{Float32}(undef, rope_pairs, cfg.max_position_embeddings)
    theta = Float32(cfg.rope_theta)
    for pair_idx in 1:rope_pairs
        i = 2 * pair_idx - 1
        freq = Float32(1.0) / (theta ^ (Float32(i - 1) / Float32(rope_dim)))
        for pos in 1:cfg.max_position_embeddings
            cache[pair_idx, pos] = cos(Float32(pos - 1) * freq)
        end
    end
    cache
end
const rope_sin_cache_gpu = oneArray(Float16.(rope_sin_cache))
const rope_cos_cache_gpu = oneArray(Float16.(rope_cos_cache))

struct SSMState
    conv::AbstractMatrix{Float16}
    h::AbstractArray{Float32, 3}

    # Scratchpads
    qkv_proj::AbstractVector{Float16}
    z_buf::AbstractVector{Float16}
    x_conv::AbstractVector{Float16}
    y_all::AbstractVector{Float16}
    branch_out::AbstractVector{Float16}
    x_norm_cpu::Vector{Float16}
    x_norm_cpu32::Vector{Float32}
    x_conv_cpu::Vector{Float16}
    y_all_cpu::Vector{Float16}
    alpha_proj::Vector{Float32}
    beta_proj::Vector{Float32}
    q_norm_buf::Vector{Float32}
    k_norm_buf::Vector{Float32}
    tmp_head::Vector{Float32}

    # Constructor
    function SSMState(config::Model.QwenConfig)
        inner_size = config.ssm_inner_size
        state_size = config.ssm_state_size
        groups = config.ssm_group_count
        num_v_heads = config.ssm_time_step_rank
        head_v_dim = inner_size ÷ num_v_heads

        # Qwen 3.5 SSM qkv_proj projects to 3 * inner_size
        qkv_proj_size = 3 * inner_size
        conv_channels = qkv_proj_size
        conv_kernel = config.ssm_conv_kernel

        new(
            oneArray{Float16}(undef, conv_channels, conv_kernel),
            zeros(Float32, head_v_dim, state_size, groups),
            oneArray{Float16}(undef, qkv_proj_size), # qkv_proj
            oneArray{Float16}(undef, inner_size), # z_buf
            oneArray{Float16}(undef, conv_channels), # x_conv
            oneArray{Float16}(undef, inner_size),     # y_all
            oneArray{Float16}(undef, config.hidden_size), # branch_out
            Vector{Float16}(undef, config.hidden_size),
            Vector{Float32}(undef, config.hidden_size),
            Vector{Float16}(undef, conv_channels),
            Vector{Float16}(undef, inner_size),
            Vector{Float32}(undef, groups),
            Vector{Float32}(undef, groups),
            Vector{Float32}(undef, state_size),
            Vector{Float32}(undef, state_size),
            Vector{Float32}(undef, head_v_dim),
        )
    end
end

struct KVCache
    k::AbstractArray{Float16, 3}
    v::AbstractArray{Float16, 3}

    # Scratchpads
    q_all::AbstractVector{Float16} # Larger buffer for both Q and Gate
    k_buf::AbstractVector{Float16}
    v_buf::AbstractVector{Float16}
    scores::AbstractVector{Float16}
    attn_out_buf::AbstractVector{Float16}
    branch_out::AbstractVector{Float16}
    mlp_gate::AbstractVector{Float16}
    mlp_up::AbstractVector{Float16}
    rope_q_tmp::AbstractMatrix{Float16}
    rope_k_tmp::AbstractMatrix{Float16}
    norm1_buf::AbstractVector{Float16}
    norm2_buf::AbstractVector{Float16}

    function KVCache(config::Model.QwenConfig)
        head_dim = config.head_dim
        n_heads_q = config.num_attention_heads
        n_heads_kv = config.num_key_value_heads
        max_pos = config.max_position_embeddings
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size

        new(
            oneArray{Float16}(undef, head_dim, n_heads_kv, max_pos),
            oneArray{Float16}(undef, head_dim, n_heads_kv, max_pos),
            oneArray{Float16}(undef, n_heads_q * head_dim * 2), # q_all (includes gate)
            oneArray{Float16}(undef, n_heads_kv * head_dim),    # k_buf
            oneArray{Float16}(undef, n_heads_kv * head_dim),    # v_buf
            oneArray{Float16}(undef, max_pos),                  # scores
            oneArray{Float16}(undef, n_heads_q * head_dim),     # attn_out_buf
            oneArray{Float16}(undef, hidden_size),              # branch_out
            oneArray{Float16}(undef, intermediate_size),        # mlp_gate
            oneArray{Float16}(undef, intermediate_size),        # mlp_up
            oneArray{Float16}(undef, rope_pairs, n_heads_q),
            oneArray{Float16}(undef, rope_pairs, n_heads_kv),
            oneArray{Float16}(undef, hidden_size),              # norm1_buf
            oneArray{Float16}(undef, hidden_size),              # norm2_buf
        )
    end
end

const ssm_caches = let cfg = runtime_config()
    [SSMState(cfg) for _ in 1:cfg.num_hidden_layers]
end
const kv_caches = let cfg = runtime_config()
    [KVCache(cfg) for _ in 1:cfg.num_hidden_layers]
end
const hidden_work_buf = let cfg = runtime_config()
    oneArray{Float16}(undef, cfg.hidden_size)
end
const hidden_out_buf = let cfg = runtime_config()
    oneArray{Float16}(undef, cfg.hidden_size)
end

function update_conv_cache!(conv, input)
    channels, kernel_size = size(conv)
    if kernel_size > 1
        @views conv[:, 1:kernel_size-1] .= conv[:, 2:kernel_size]
    end
    @views conv[:, kernel_size] .= input
end

function update_kv_cache!(cache, k, v, pos)
    @views cache.k[:, :, pos] .= k
    @views cache.v[:, :, pos] .= v
end

function apply_rope!(q, k, pos, config, kv_cache)
    sin_vals = reshape(@view(rope_sin_cache_gpu[:, pos]), rope_pairs, 1)
    cos_vals = reshape(@view(rope_cos_cache_gpu[:, pos]), rope_pairs, 1)

    q_odd = @view q[1:2:rope_dim, :]
    q_even = @view q[2:2:rope_dim, :]
    copyto!(kv_cache.rope_q_tmp, q_odd)
    @. q_odd = kv_cache.rope_q_tmp * cos_vals - q_even * sin_vals
    @. q_even = kv_cache.rope_q_tmp * sin_vals + q_even * cos_vals

    k_odd = @view k[1:2:rope_dim, :]
    k_even = @view k[2:2:rope_dim, :]
    copyto!(kv_cache.rope_k_tmp, k_odd)
    @. k_odd = kv_cache.rope_k_tmp * cos_vals - k_even * sin_vals
    @. k_even = kv_cache.rope_k_tmp * sin_vals + k_even * cos_vals
end

function process_branch!(::Val{:ssm}, block::LoadedBlock, x_norm, state_cache, config)
    inner_size = config.ssm_inner_size
    state_size = config.ssm_state_size
    groups = config.ssm_group_count
    head_v_dim = inner_size ÷ config.ssm_time_step_rank
    T = eltype(x_norm)

    mul!(state_cache.qkv_proj, block.attn_qkv_weight, x_norm)
    mul!(state_cache.z_buf, block.attn_gate_weight, x_norm)

    update_conv_cache!(state_cache.conv, state_cache.qkv_proj)
    fill!(state_cache.x_conv, T(0))
    for k in 1:config.ssm_conv_kernel
        @views @. state_cache.x_conv += state_cache.conv[:, k] * block.ssm_conv1d_weight[:, k]
    end

    # SiLU on convolved input
    @. state_cache.x_conv = state_cache.x_conv * (T(1.0) / (T(1.0) + exp(-state_cache.x_conv)))

    copyto!(state_cache.x_conv_cpu, state_cache.x_conv)
    copyto!(state_cache.x_norm_cpu, x_norm)
    fill!(state_cache.y_all_cpu, T(0))
    @inbounds for i in eachindex(state_cache.x_norm_cpu)
        state_cache.x_norm_cpu32[i] = Float32(state_cache.x_norm_cpu[i])
    end

    qk_size = state_size * groups
    q_all = reshape(@view(state_cache.x_conv_cpu[1:qk_size]), state_size, groups)
    k_all = reshape(@view(state_cache.x_conv_cpu[qk_size+1:2*qk_size]), state_size, groups)
    v_all = reshape(@view(state_cache.x_conv_cpu[2*qk_size+1:2*qk_size+inner_size]), head_v_dim, groups)

    x_norm_cpu32 = state_cache.x_norm_cpu32
    y_all_cpu = state_cache.y_all_cpu
    alpha_weight_cpu = block.ssm_alpha_weight_cpu
    beta_weight_cpu = block.ssm_beta_weight_cpu
    dt_bias_cpu = block.ssm_dt_bias_cpu
    ssm_a_cpu = block.ssm_a_cpu
    alpha_proj = state_cache.alpha_proj
    beta_proj = state_cache.beta_proj
    q_norm_buf = state_cache.q_norm_buf
    k_norm_buf = state_cache.k_norm_buf
    tmp_head = state_cache.tmp_head

    mul!(alpha_proj, alpha_weight_cpu, x_norm_cpu32)
    mul!(beta_proj, beta_weight_cpu, x_norm_cpu32)

    for g in 1:groups
        qg = view(q_all, :, g)
        kg = view(k_all, :, g)
        vg = view(v_all, :, g)

        q_norm_sq = mapreduce(v -> Float32(v)^2, +, qg)
        k_norm_sq = mapreduce(v -> Float32(v)^2, +, kg)
        q_norm_val = T(sqrt(q_norm_sq + Float32(config.rms_norm_eps)))
        k_norm_val = T(sqrt(k_norm_sq + Float32(config.rms_norm_eps)))

        inv_q_norm = Float32(1) / Float32(q_norm_val)
        inv_k_norm = Float32(1) / Float32(k_norm_val)
        @inbounds @simd for j in 1:state_size
            q_norm_buf[j] = Float32(qg[j]) * inv_q_norm
            k_norm_buf[j] = Float32(kg[j]) * inv_k_norm
        end

        dg = Float32(exp(log(1.0 + exp(alpha_proj[g] + dt_bias_cpu[g])) * ssm_a_cpu[g]))
        bg = Float32(1.0 / (1.0 + exp(-beta_proj[g])))

        state = view(state_cache.h, :, :, g)
        state .*= dg

        mul!(tmp_head, state, k_norm_buf)
        @inbounds @simd for i in 1:head_v_dim
            tmp_head[i] = bg * (Float32(vg[i]) - tmp_head[i])
        end
        BLAS.ger!(1.0f0, tmp_head, k_norm_buf, state)

        yg = view(y_all_cpu, (g-1)*head_v_dim+1:g*head_v_dim)
        mul!(tmp_head, state, q_norm_buf)
        @inbounds @simd for i in 1:head_v_dim
            yg[i] = T(tmp_head[i])
        end
    end

    # Copy back to GPU
    copyto!(state_cache.y_all, y_all_cpu)
    y_all = state_cache.y_all

    y_all_reshaped = reshape(y_all, head_v_dim, groups)
    Model.rmsnorm!(y_all_reshaped, y_all_reshaped, block.ssm_norm)

    @. state_cache.z_buf = state_cache.z_buf * (T(1.0) / (T(1.0) + exp(-state_cache.z_buf)))
    @. y_all *= state_cache.z_buf
    mul!(state_cache.branch_out, block.ssm_out_weight, y_all)

    # CPU Float32 fallback for NaN/Inf overflow
    branch_out_cpu = Float32.(Array(state_cache.branch_out))
    branch_out_rms = sqrt(sum(abs2, branch_out_cpu) / length(branch_out_cpu))
    if !all(isfinite, branch_out_cpu) || branch_out_rms > 4.0f0
        eps = Float32(config.rms_norm_eps)
        x_norm_cpu = Float32.(Array(x_norm))

        qkv_w = Float32.(Array(block.attn_qkv_weight))
        gate_w = Float32.(Array(block.attn_gate_weight))
        qkv_proj = qkv_w * x_norm_cpu
        z_buf_cpu = gate_w * x_norm_cpu

        conv_channels = 2 * groups * state_size + inner_size
        K = config.ssm_conv_kernel
        conv_state_cpu = Float32.(Array(state_cache.conv))
        if K > 1; conv_state_cpu[:, 1:K-1] = conv_state_cpu[:, 2:K]; end
        conv_state_cpu[:, K] = qkv_proj
        copyto!(state_cache.conv, Float16.(conv_state_cpu))

        x_conv = zeros(Float32, conv_channels)
        conv1d_w = Float32.(Array(block.ssm_conv1d_weight))
        for k in 1:K; x_conv .+= conv_state_cpu[:, k] .* conv1d_w[:, k]; end
        @. x_conv = x_conv * (1.0f0 / (1.0f0 + exp(-x_conv)))

        qk_size = state_size * groups
        q_all = reshape(x_conv[1:qk_size], state_size, groups)
        k_all = reshape(x_conv[qk_size+1:2*qk_size], state_size, groups)
        v_all = reshape(x_conv[2*qk_size+1:2*qk_size+inner_size], head_v_dim, groups)

        alpha_proj = block.ssm_alpha_weight_cpu * x_norm_cpu
        beta_proj = block.ssm_beta_weight_cpu * x_norm_cpu

        y_all_cpu = zeros(Float32, inner_size)
        for g in 1:groups
            qg = view(q_all, :, g); kg = view(k_all, :, g); vg = view(v_all, :, g)
            qn = sqrt(sum(v -> Float32(v)^2, qg) + eps)
            kn = sqrt(sum(v -> Float32(v)^2, kg) + eps)
            qn_buf = Float32.(qg) ./ qn; kn_buf = Float32.(kg) ./ kn
            dg = Float32(exp(log(1.0 + exp(Float64(alpha_proj[g]) + Float64(block.ssm_dt_bias_cpu[g]))) * Float64(block.ssm_a_cpu[g])))
            bg = Float32(1.0 / (1.0 + exp(-Float64(beta_proj[g]))))
            state = view(state_cache.h, :, :, g)
            state .*= dg
            sk = state * kn_buf
            @inbounds for i in eachindex(sk); sk[i] = bg * (Float32(vg[i]) - sk[i]); end
            BLAS.ger!(1.0f0, sk, kn_buf, state)
            yg = view(y_all_cpu, (g-1)*head_v_dim+1:g*head_v_dim)
            tmp = state * qn_buf
            @inbounds for i in eachindex(tmp); yg[i] = tmp[i]; end
        end
        copyto!(state_cache.y_all, Float16.(y_all_cpu))

        # SSM norm + gate + output on CPU
        y_reshaped = reshape(y_all_cpu, head_v_dim, groups)
        ssm_norm_w = Float32.(Array(block.ssm_norm.weight))
        for g in 1:groups
            v = view(y_reshaped, :, g)
            sc = 1.0f0 / sqrt(sum(v -> Float32(v)^2, v) / head_v_dim + eps)
            @inbounds for i in 1:head_v_dim; v[i] *= sc * ssm_norm_w[i]; end
        end
        @. z_buf_cpu = z_buf_cpu * (1.0f0 / (1.0f0 + exp(-z_buf_cpu)))
        y_reshaped .*= reshape(z_buf_cpu, head_v_dim, groups)
        ssm_out_w = Float32.(Array(block.ssm_out_weight))
        branch_out_cpu = ssm_out_w * vec(y_reshaped)
        copyto!(state_cache.branch_out, Float16.(branch_out_cpu))

        warned = _mlp_cpu_warned()
        if block.index ∉ warned
            push!(warned, block.index)
            @warn "Falling back to CPU Float32 SSM for unstable layer" layer=block.index gpu_rms=branch_out_rms cpu_rms=sqrt(sum(abs2, branch_out_cpu) / length(branch_out_cpu))
        end
    end
    return state_cache.branch_out
end

function process_branch!(::Val{:attn}, block::LoadedBlock, x_norm, kv_cache, config, pos)
    head_dim = config.head_dim
    n_heads_q = config.num_attention_heads
    n_heads_kv = config.num_key_value_heads
    T = eltype(x_norm)

    q_all_buf = kv_cache.q_all
    mul!(q_all_buf, block.attn_q_weight, x_norm)

    k_buf = kv_cache.k_buf
    mul!(k_buf, block.attn_k_weight, x_norm)

    v_buf = kv_cache.v_buf
    mul!(v_buf, block.attn_v_weight, x_norm)

    q_size = n_heads_q * head_dim
    q = view(q_all_buf, 1:q_size)
    attn_gate = view(q_all_buf, q_size+1:2*q_size)

    q_heads_raw = reshape(q, head_dim, n_heads_q)
    k_heads_raw = reshape(k_buf, head_dim, n_heads_kv)

    Model.rmsnorm!(q_heads_raw, q_heads_raw, block.attn_q_norm)
    Model.rmsnorm!(k_heads_raw, k_heads_raw, block.attn_k_norm)

    q_heads = q_heads_raw
    k_heads = k_heads_raw
    v_heads = reshape(v_buf, head_dim, n_heads_kv)

    @. attn_gate = attn_gate * (T(1.0) / (T(1.0) + exp(-attn_gate)))
    q_heads .*= reshape(attn_gate, head_dim, n_heads_q)

    apply_rope!(q_heads, k_heads, pos, config, kv_cache)
    update_kv_cache!(kv_cache, k_heads, v_heads, pos)

    attn_out = reshape(kv_cache.attn_out_buf, head_dim, n_heads_q)
    scale = T(1.0) / sqrt(T(head_dim))
    gqa_ratio = n_heads_q ÷ n_heads_kv

    for h in 1:n_heads_q
        kv_h = ((h - 1) ÷ gqa_ratio) + 1
        K_past = view(kv_cache.k, :, kv_h, 1:pos)
        V_past = view(kv_cache.v, :, kv_h, 1:pos)

        scores = view(kv_cache.scores, 1:pos)
        mul!(scores, K_past', view(q_heads, :, h))
        scores .*= scale

        mx = maximum(scores)
        @. scores = exp(scores - mx)
        scores ./= sum(scores)

        mul!(view(attn_out, :, h), V_past, scores)
    end

    mul!(kv_cache.branch_out, block.attn_output_weight, vec(attn_out))

    # CPU Float32 fallback for NaN/Inf overflow
    branch_out_cpu = Float32.(Array(kv_cache.branch_out))
    branch_out_rms = sqrt(sum(abs2, branch_out_cpu) / length(branch_out_cpu))
    if !all(isfinite, branch_out_cpu) || branch_out_rms > 4.0f0
        x_norm_cpu = Float32.(Array(x_norm))
        eps = Float32(config.rms_norm_eps)

        # CPU Q/K/V projections
        q_all = Float32.(Array(block.attn_q_weight)) * x_norm_cpu
        k_buf = Float32.(Array(block.attn_k_weight)) * x_norm_cpu
        v_buf = Float32.(Array(block.attn_v_weight)) * x_norm_cpu

        q_size = n_heads_q * head_dim
        q_cpu = q_all[1:q_size]
        gate_cpu = q_all[q_size+1:2*q_size]
        q_heads = reshape(q_cpu, head_dim, n_heads_q)
        k_heads = reshape(k_buf, head_dim, n_heads_kv)
        v_heads = reshape(v_buf, head_dim, n_heads_kv)

        # CPU Q/K norm
        q_norm_w = Float32.(Array(block.attn_q_norm.weight))
        k_norm_w = Float32.(Array(block.attn_k_norm.weight))
        for h in 1:n_heads_q
            v = view(q_heads, :, h)
            sc = 1.0f0 / sqrt(sum(v -> Float32(v)^2, v) / head_dim + eps)
            @inbounds for i in 1:head_dim; v[i] *= sc * q_norm_w[i]; end
        end
        for h in 1:n_heads_kv
            v = view(k_heads, :, h)
            sc = 1.0f0 / sqrt(sum(v -> Float32(v)^2, v) / head_dim + eps)
            @inbounds for i in 1:head_dim; v[i] *= sc * k_norm_w[i]; end
        end

        # CPU SiLU gate * Q
        gate_reshaped = reshape(gate_cpu, head_dim, n_heads_q)
        @. gate_reshaped = gate_reshaped * (1.0f0 / (1.0f0 + exp(-gate_reshaped)))
        q_heads .*= gate_reshaped

        # CPU RoPE
        rope_dim = head_dim
        rope_pairs = rope_dim ÷ 2
        for h in 1:n_heads_q
            for p in 1:rope_pairs
                i0 = 2p - 1; i1 = 2p
                s = Float64(rope_sin_cache[p, pos]); c = Float64(rope_cos_cache[p, pos])
                q0 = Float64(q_heads[i0, h]); q1 = Float64(q_heads[i1, h])
                q_heads[i0, h] = Float32(q0 * c - q1 * s)
                q_heads[i1, h] = Float32(q0 * s + q1 * c)
            end
        end
        for h in 1:n_heads_kv
            for p in 1:rope_pairs
                i0 = 2p - 1; i1 = 2p
                s = Float64(rope_sin_cache[p, pos]); c = Float64(rope_cos_cache[p, pos])
                k0 = Float64(k_heads[i0, h]); k1 = Float64(k_heads[i1, h])
                k_heads[i0, h] = Float32(k0 * c - k1 * s)
                k_heads[i1, h] = Float32(k0 * s + k1 * c)
            end
        end

        # CPU attention
        attn_out_cpu = zeros(Float32, head_dim, n_heads_q)
        scale = Float32(1.0 / sqrt(head_dim))
        for h in 1:n_heads_q
            kv_h = ((h - 1) ÷ gqa_ratio) + 1
            K_past = Float64.(Array(kv_cache.k[:, kv_h, 1:pos]))
            V_past = Float64.(Array(kv_cache.v[:, kv_h, 1:pos]))
            q_h = Float64.(q_heads[:, h])
            scores = K_past' * q_h
            scores .*= Float64(scale)
            scores .-= maximum(scores)
            scores = exp.(scores)
            scores ./= sum(scores)
            attn_out_cpu[:, h] = Float32.(V_past * scores)
        end

        attn_out_w = Float32.(Array(block.attn_output_weight))
        branch_out_cpu = attn_out_w * vec(attn_out_cpu)
        copyto!(kv_cache.branch_out, Float16.(branch_out_cpu))

        warned = _mlp_cpu_warned()
        if block.index ∉ warned
            push!(warned, block.index)
            @warn "Falling back to CPU Float32 attention for unstable layer" layer=block.index gpu_rms=branch_out_rms cpu_rms=sqrt(sum(abs2, branch_out_cpu) / length(branch_out_cpu))
        end
    end
    return kv_cache.branch_out
end

function process_mlp!(block::LoadedBlock, x_norm, cache)
    T = eltype(x_norm)
    mul!(cache.mlp_gate, block.ffn_gate_weight, x_norm)
    mul!(cache.mlp_up, block.ffn_up_weight, x_norm)
    @. cache.mlp_gate = cache.mlp_gate * (T(1.0) / (T(1.0) + exp(-cache.mlp_gate)))
    cache.mlp_gate .*= cache.mlp_up
    mul!(cache.branch_out, block.ffn_down_weight, cache.mlp_gate)
    branch_out_cpu = Float32.(Array(cache.branch_out))
    branch_out_rms = sqrt(sum(abs2, branch_out_cpu) / length(branch_out_cpu))
    if !all(isfinite, branch_out_cpu) || branch_out_rms > 4.0f0
        cache_dict = _mlp_cpu_cache()
        weights = get!(cache_dict, block.index) do
            (
                gate=Float32.(Array(block.ffn_gate_weight)),
                up=Float32.(Array(block.ffn_up_weight)),
                down=Float32.(Array(block.ffn_down_weight)),
            )
        end
        x_norm_cpu = Float32.(Array(x_norm))
        gate_cpu = weights.gate * x_norm_cpu
        up_cpu = weights.up * x_norm_cpu
        @. gate_cpu = gate_cpu * (1.0f0 / (1.0f0 + exp(-gate_cpu)))
        gate_cpu .*= up_cpu
        branch_out_cpu = weights.down * gate_cpu
        copyto!(cache.branch_out, Float16.(branch_out_cpu))
        warned = _mlp_cpu_warned()
        if block.index ∉ warned
            push!(warned, block.index)
            @warn "Falling back to CPU Float32 MLP for unstable layer" layer=block.index gpu_rms=branch_out_rms cpu_rms=sqrt(sum(abs2, branch_out_cpu) / length(branch_out_cpu))
        end
    end
    return cache.branch_out
end




# Runtime wrappers have been moved to cleanup_memory.jl

# Trace helpers moved to inference_trace.jl

function forward_hidden_core!(x, loaded_blocks, config, file, pos)
    ensure_runtime_available!()
    if pos == 1
        reset_caches!()
    end

    for i in 1:config.num_hidden_layers
        blk = loaded_blocks[i]
        cache = kv_caches[i]

        # Branch 1
        x_norm1 = cache.norm1_buf
        Model.rmsnorm!(x_norm1, x, blk.attn_norm)
        if blk.is_ssm
            x_branch = process_branch!(Val(:ssm), blk, x_norm1, ssm_caches[i], config)
        else
            x_branch = process_branch!(Val(:attn), blk, x_norm1, cache, config, pos)
        end
        x .+= x_branch

        # Branch 2
        x_norm2 = cache.norm2_buf
        Model.rmsnorm!(x_norm2, x, blk.post_attn_norm)
        x_mlp = process_mlp!(blk, x_norm2, cache) # Reuse kv_cache for MLP buffers
        x .+= x_mlp
    end

    return x
end

function forward_hidden!(x, loaded_blocks, config, file, pos)
    ensure_runtime_available!()
    forward_hidden_core!(x, loaded_blocks, config, file, pos)
    return Model.rmsnorm!(hidden_out_buf, x, final_rms_pre)
end

function forward_hidden(toks, loaded_blocks, config, file, pos)
    x = load_hidden!(hidden_work_buf, toks)
    return forward_hidden!(x, loaded_blocks, config, file, pos)
end

function forward_token_id!(tok_id::Integer, pos::Integer)
    ensure_runtime_available!()
    @views copyto!(hidden_work_buf, embed_gpu[tok_id, :])
    return forward_hidden!(hidden_work_buf, loaded_blocks, config, file, pos)
end

function prefill_token_id!(tok_id::Integer, pos::Integer, last_pos::Integer)
    ensure_runtime_available!()
    @views copyto!(hidden_work_buf, embed_gpu[tok_id, :])
    if pos == last_pos
        return forward_hidden!(hidden_work_buf, loaded_blocks, config, file, pos)
    end
    forward_hidden_core!(hidden_work_buf, loaded_blocks, config, file, pos)
    return nothing
end

function project_to_logits(x)
    ensure_runtime_available!()
    return lm_head_weight * x
end

function project_to_logits!(dest, x)
    ensure_runtime_available!()
    return mul!(dest, lm_head_weight, x)
end

function greedy_argmax(logits)
    ensure_runtime_available!()
    copyto!(logits_host, logits)
    return argmax(logits_host)
end

function greedy_argmax_hidden(hidden)
    ensure_runtime_available!()
    project_to_logits!(logits_work_buf, hidden)
    copyto!(logits_host, logits_work_buf)
    return argmax(logits_host)
end

function forward_pass(toks, loaded_blocks, config, file, pos)
    return forward_hidden(toks, loaded_blocks, config, file, pos)
end

function default_chat_prompt(prompt::AbstractString; system_prompt::Union{Nothing, AbstractString}=nothing)
    return render_chat_template(
        [(; role="user", content=String(prompt))];
        system_prompt=system_prompt,
        add_generation_prompt=true,
        enable_thinking=false,
    )
end

function _normalize_message_content(content)
    content === nothing && return ""
    if content isa AbstractString
        return String(content)
    end
    if content isa AbstractDict
        if haskey(content, :text)
            return String(content[:text])
        elseif haskey(content, "text")
            return String(content["text"])
        end
    elseif content isa AbstractVector || content isa Tuple
        parts = String[]
        for item in content
            if item isa AbstractString
                push!(parts, String(item))
            elseif item isa AbstractDict
                item_type = if haskey(item, :type)
                    item[:type]
                elseif haskey(item, "type")
                    item["type"]
                else
                    nothing
                end
                if item_type == "text"
                    if haskey(item, :text)
                        push!(parts, String(item[:text]))
                    elseif haskey(item, "text")
                        push!(parts, String(item["text"]))
                    else
                        error("Text chat content items must define a text field")
                    end
                else
                    error("Only text chat content items are supported by TestInference chat rendering")
                end
            else
                error("Only text chat content items are supported by TestInference chat rendering")
            end
        end
        return join(parts)
    end
    error("Only plain-text message content is supported by TestInference chat rendering")
end

function _message_field(msg, name::Symbol, default=nothing)
    if hasproperty(msg, name)
        return getproperty(msg, name)
    elseif msg isa AbstractDict
        key = String(name)
        if haskey(msg, name)
            return msg[name]
        elseif haskey(msg, key)
            return msg[key]
        end
    end
    return default
end

function _message_role(msg)
    role = _message_field(msg, :role, nothing)
    role === nothing && error("Chat messages must define a role")
    return String(role)
end

function _normalize_messages(messages; system_prompt::Union{Nothing, AbstractString}=nothing)
    normalized = Vector{NamedTuple{(:role, :content, :tool_calls), Tuple{String, String, Any}}}()

    if system_prompt !== nothing && !isempty(system_prompt)
        push!(normalized, (; role="system", content=String(system_prompt), tool_calls=nothing))
    end

    for msg in messages
        push!(
            normalized,
            (
                role=_message_role(msg),
                content=_normalize_message_content(_message_field(msg, :content, "")),
                tool_calls=_message_field(msg, :tool_calls, nothing),
            ),
        )
    end

    return normalized
end

function _template_contains_think_tag(template::Union{Nothing, AbstractString})
    template !== nothing && occursin("<think>", template)
end

function _ensure_supported_chat_template(template::Union{Nothing, AbstractString}, messages)
    template === nothing && return :fallback

    occursin("<|im_start|>", template) || error("Unsupported chat template: missing <|im_start|> marker")
    occursin("<|im_end|>", template) || error("Unsupported chat template: missing <|im_end|> marker")
    occursin("add_generation_prompt", template) || error("Unsupported chat template: missing generation-prompt logic")

    for msg in messages
        msg.tool_calls === nothing || isempty(msg.tool_calls) ||
            error("Tool-call rendering is not supported by TestInference chat rendering")
    end

    return :qwen_chatml
end

function render_chat_template(
    messages;
    system_prompt::Union{Nothing, AbstractString}=nothing,
    add_generation_prompt::Bool=true,
    enable_thinking::Bool=false,
)
    normalized = _normalize_messages(messages; system_prompt=system_prompt)
    template = get_chat_template()
    template_kind = _ensure_supported_chat_template(template, normalized)
    parts = String[]

    if template_kind == :fallback
        for msg in normalized
            if msg.role == "system"
                push!(parts, "<|im_start|>system\n$(msg.content)<|im_end|>\n")
            elseif msg.role == "user"
                push!(parts, "<|im_start|>user\n$(msg.content)<|im_end|>\n")
            elseif msg.role == "assistant"
                push!(parts, "<|im_start|>assistant\n$(msg.content)<|im_end|>\n")
            elseif msg.role == "tool"
                push!(parts, "<|im_start|>user\n<tool_response>\n$(msg.content)\n</tool_response><|im_end|>\n")
            else
                error("Unexpected message role: $(msg.role)")
            end
        end
    else
        for i in eachindex(normalized)
            msg = normalized[i]
            role = msg.role

            if role == "system"
                push!(parts, "<|im_start|>system\n$(msg.content)<|im_end|>\n")
            elseif role == "user"
                push!(parts, "<|im_start|>user\n$(msg.content)<|im_end|>\n")
            elseif role == "assistant"
                push!(parts, "<|im_start|>assistant\n$(msg.content)<|im_end|>\n")
            elseif role == "tool"
                prev_is_tool = i > firstindex(normalized) && normalized[i - 1].role == "tool"
                next_is_tool = i < lastindex(normalized) && normalized[i + 1].role == "tool"
                if !prev_is_tool
                    push!(parts, "<|im_start|>user\n")
                end
                push!(parts, "<tool_response>\n$(msg.content)\n</tool_response>")
                if !next_is_tool
                    push!(parts, "<|im_end|>\n")
                end
            else
                error("Unexpected message role: $role")
            end
        end
    end

    if add_generation_prompt
        push!(parts, "<|im_start|>assistant\n")
        if _template_contains_think_tag(template)
            if enable_thinking
                push!(parts, "<think>\n")
            else
                push!(parts, "<think>\n\n</think>\n\n")
            end
        end
    end

    return join(parts)
end

function format_chat_prompt(prompt::AbstractString; system_prompt::Union{Nothing, AbstractString}=nothing, enable_thinking::Bool=false)
    return render_chat_template(
        [(; role="user", content=String(prompt))];
        system_prompt=system_prompt,
        add_generation_prompt=true,
        enable_thinking=enable_thinking,
    )
end

function encode_prompt(
    prompt::AbstractString;
    add_bos::Bool=false,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
)
    rendered_prompt = use_chat_template ? format_chat_prompt(prompt; system_prompt=system_prompt, enable_thinking=enable_thinking) : String(prompt)
    ids = Inferno.Tokenizer.encode(tokenizer, rendered_prompt)
    if add_bos && tokenizer.bos_id > 0
        return vcat([tokenizer.bos_id], ids)
    end
    return ids
end

function run_prompt(
    prompt::AbstractString;
    add_bos::Bool=false,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(getfield(latest, :run_prompt), prompt; add_bos=add_bos, use_chat_template=use_chat_template, system_prompt=system_prompt, enable_thinking=enable_thinking)
    end
    token_ids = encode_prompt(
        prompt;
        add_bos=add_bos,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    isempty(token_ids) && error("Prompt encoded to zero tokens")

    hidden = prefill_prompt_tokens(token_ids)
    logits = project_to_logits(hidden)
    return (; token_ids, hidden, logits)
end

function prefill_prompt_tokens(token_ids)
    isempty(token_ids) && error("Prompt encoded to zero tokens")

    last_pos = length(token_ids)
    hidden = nothing
    for (pos, tok_id) in enumerate(token_ids)
        hidden = prefill_token_id!(tok_id, pos, last_pos)
    end

    return hidden
end

function next_token(
    prompt::AbstractString;
    add_bos::Bool=false,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(getfield(latest, :next_token), prompt; add_bos=add_bos, use_chat_template=use_chat_template, system_prompt=system_prompt, enable_thinking=enable_thinking)
    end
    result = run_prompt(
        prompt;
        add_bos=add_bos,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    token_id = greedy_argmax(result.logits)
    text = Inferno.Tokenizer.decode(tokenizer, [token_id])
    return (; token_id, text, logits=result.logits, hidden=result.hidden, prompt_token_ids=result.token_ids)
end

function generate(
    prompt::AbstractString;
    max_new_tokens::Int=32,
    add_bos::Bool=false,
    stop_at_eos::Bool=true,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(getfield(latest, :generate), prompt; max_new_tokens=max_new_tokens, add_bos=add_bos, stop_at_eos=stop_at_eos, use_chat_template=use_chat_template, system_prompt=system_prompt, enable_thinking=enable_thinking)
    end
    max_new_tokens >= 0 || error("max_new_tokens must be non-negative")

    prompt_token_ids = encode_prompt(
        prompt;
        add_bos=add_bos,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    isempty(prompt_token_ids) && error("Prompt encoded to zero tokens")

    generated_ids = Int[]
    hidden = prefill_prompt_tokens(prompt_token_ids)

    has_eos = tokenizer.eos_id > 0
    current_pos = length(prompt_token_ids)
    for _ in 1:max_new_tokens
        next_id = greedy_argmax_hidden(hidden)
        push!(generated_ids, next_id)

        if stop_at_eos && has_eos && next_id == tokenizer.eos_id
            break
        end

        current_pos += 1
        hidden = forward_token_id!(next_id, current_pos)
    end

    generated_text = Inferno.Tokenizer.decode(tokenizer, generated_ids)
    return (; prompt_token_ids, generated_ids, generated_text)
end

function generate_tokens!(
    on_token,
    prompt::AbstractString;
    max_new_tokens::Int=32,
    add_bos::Bool=false,
    stop_at_eos::Bool=true,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(getfield(latest, :generate_tokens!), on_token, prompt; max_new_tokens=max_new_tokens, add_bos=add_bos, stop_at_eos=stop_at_eos, use_chat_template=use_chat_template, system_prompt=system_prompt, enable_thinking=enable_thinking)
    end
    max_new_tokens >= 0 || error("max_new_tokens must be non-negative")

    prompt_token_ids = encode_prompt(
        prompt;
        add_bos=add_bos,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    isempty(prompt_token_ids) && error("Prompt encoded to zero tokens")

    generated_ids = Int[]
    hidden = prefill_prompt_tokens(prompt_token_ids)

    current_pos = length(prompt_token_ids)
    for step in 1:max_new_tokens
        project_to_logits!(logits_work_buf, hidden)
        next_id = greedy_argmax(logits_work_buf)
        push!(generated_ids, next_id)

        token_text = Inferno.Tokenizer.decode(tokenizer, [next_id])
        on_token((; step, token_id=next_id, text=token_text, logits=logits_work_buf, hidden, position=current_pos))

        if stop_at_eos && tokenizer.eos_id > 0 && next_id == tokenizer.eos_id
            break
        end

        current_pos += 1
        hidden = forward_token_id!(next_id, current_pos)
    end

    generated_text = Inferno.Tokenizer.decode(tokenizer, generated_ids)
    return (; prompt_token_ids, generated_ids, generated_text)
end

function generate_stream(
    prompt::AbstractString;
    max_new_tokens::Int=32,
    add_bos::Bool=false,
    stop_at_eos::Bool=true,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(getfield(latest, :generate_stream), prompt; max_new_tokens=max_new_tokens, add_bos=add_bos, stop_at_eos=stop_at_eos, use_chat_template=use_chat_template, system_prompt=system_prompt, enable_thinking=enable_thinking)
    end
    ensure_runtime_available!()
    return Channel{String}(32) do chan
        try
            generate_tokens!(
                token -> put!(chan, token.text),
                prompt;
                max_new_tokens=max_new_tokens,
                add_bos=add_bos,
                stop_at_eos=stop_at_eos,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
            )
        catch e
            if !(e isa InterruptException || e isa InvalidStateException)
                @error "ERROR during TestInference generation stream" exception=(e, catch_backtrace())
            end
        finally
            close(chan)
        end
    end
end

function stream_to_stdout(
    prompt::AbstractString;
    max_new_tokens::Int=32,
    add_bos::Bool=false,
    stop_at_eos::Bool=true,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
    io::IO=stdout,
)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(getfield(latest, :stream_to_stdout), prompt; max_new_tokens=max_new_tokens, add_bos=add_bos, stop_at_eos=stop_at_eos, use_chat_template=use_chat_template, system_prompt=system_prompt, enable_thinking=enable_thinking, io=io)
    end
    ensure_runtime_available!()
    stream = generate_stream(
        prompt;
        max_new_tokens=max_new_tokens,
        add_bos=add_bos,
        stop_at_eos=stop_at_eos,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )

    generated_text = IOBuffer()
    for token in stream
        print(io, token)
        flush(io)
        print(generated_text, token)
    end
    println(io)
    flush(io)

    return String(take!(generated_text))
end

function apply_rope_reference_cpu!(q_cpu, k_cpu, pos, config)
    T = eltype(q_cpu)
    local_rope_dim = config.qk_rope_head_dim > 0 ? config.qk_rope_head_dim : config.head_dim
    theta = Float32(config.rope_theta)
    p = Float32(pos - 1)

    for i in 1:2:local_rope_dim
        freq = Float32(1.0) / (theta ^ (Float32(i - 1) / Float32(local_rope_dim)))
        sin_val, cos_val = sincos(p * freq)
        for h in 1:size(q_cpu, 2)
            v1 = Float32(q_cpu[i, h])
            v2 = Float32(q_cpu[i + 1, h])
            q_cpu[i, h] = T(v1 * cos_val - v2 * sin_val)
            q_cpu[i + 1, h] = T(v1 * sin_val + v2 * cos_val)
        end
        for h in 1:size(k_cpu, 2)
            v1 = Float32(k_cpu[i, h])
            v2 = Float32(k_cpu[i + 1, h])
            k_cpu[i, h] = T(v1 * cos_val - v2 * sin_val)
            k_cpu[i + 1, h] = T(v1 * sin_val + v2 * cos_val)
        end
    end

    return q_cpu, k_cpu
end

function process_branch_ssm_reference!(block::LoadedBlock, x_norm, state_cache, config)
    inner_size = config.ssm_inner_size
    state_size = config.ssm_state_size
    groups = config.ssm_group_count
    head_v_dim = inner_size ÷ config.ssm_time_step_rank
    T = eltype(x_norm)

    mul!(state_cache.qkv_proj, block.attn_qkv_weight, x_norm)
    mul!(state_cache.z_buf, block.attn_gate_weight, x_norm)

    update_conv_cache!(state_cache.conv, state_cache.qkv_proj)
    fill!(state_cache.x_conv, T(0))
    for k in 1:config.ssm_conv_kernel
        @views @. state_cache.x_conv += state_cache.conv[:, k] * block.ssm_conv1d_weight[:, k]
    end
    @. state_cache.x_conv = state_cache.x_conv * (T(1.0) / (T(1.0) + exp(-state_cache.x_conv)))

    x_conv_cpu = Array(state_cache.x_conv)
    x_norm_cpu = Array(x_norm)
    y_all_cpu = zeros(T, inner_size)

    qk_size = state_size * groups
    q_all = reshape(view(x_conv_cpu, 1:qk_size), state_size, groups)
    k_all = reshape(view(x_conv_cpu, qk_size+1:2*qk_size), state_size, groups)
    v_all = reshape(view(x_conv_cpu, 2*qk_size+1:2*qk_size+inner_size), head_v_dim, groups)

    alpha_weight_cpu = block.ssm_alpha_weight_cpu
    beta_weight_cpu = block.ssm_beta_weight_cpu
    dt_bias_cpu = block.ssm_dt_bias_cpu
    ssm_a_cpu = block.ssm_a_cpu

    for g in 1:groups
        qg = view(q_all, :, g)
        kg = view(k_all, :, g)
        vg = view(v_all, :, g)

        q_norm_sq = mapreduce(v -> Float32(v)^2, +, qg)
        k_norm_sq = mapreduce(v -> Float32(v)^2, +, kg)
        q_norm_val = T(sqrt(q_norm_sq + Float32(config.rms_norm_eps)))
        k_norm_val = T(sqrt(k_norm_sq + Float32(config.rms_norm_eps)))

        alpha_g = dot(view(alpha_weight_cpu, g, :), Float32.(x_norm_cpu))
        beta_g = dot(view(beta_weight_cpu, g, :), Float32.(x_norm_cpu))
        dg = Float32(exp(log(1.0 + exp(alpha_g + dt_bias_cpu[g])) * ssm_a_cpu[g]))
        bg = Float32(1.0 / (1.0 + exp(-beta_g)))

        state = view(state_cache.h, :, :, g)
        state .*= dg

        for i in 1:head_v_dim
            sk_i = 0.0f0
            for j in 1:state_size
                sk_i += state[i, j] * Float32(kg[j] / k_norm_val)
            end
            update_val = bg * (Float32(vg[i]) - sk_i)
            for j in 1:state_size
                state[i, j] += update_val * Float32(kg[j] / k_norm_val)
            end
        end

        yg = view(y_all_cpu, (g-1)*head_v_dim+1:g*head_v_dim)
        for i in 1:head_v_dim
            s = 0.0f0
            for j in 1:state_size
                s += state[i, j] * Float32(qg[j] / q_norm_val)
            end
            yg[i] = T(s)
        end
    end

    copyto!(state_cache.y_all, y_all_cpu)
    y_all = state_cache.y_all
    y_all_reshaped = reshape(y_all, head_v_dim, groups)
    Model.rmsnorm!(y_all_reshaped, y_all_reshaped, block.ssm_norm)
    @. state_cache.z_buf = state_cache.z_buf * (T(1.0) / (T(1.0) + exp(-state_cache.z_buf)))
    @. y_all *= state_cache.z_buf
    mul!(state_cache.branch_out, block.ssm_out_weight, y_all)
    return state_cache.branch_out
end

end # module
