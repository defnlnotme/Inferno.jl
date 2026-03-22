# Memory cleanup and runtime management helpers for inference

if !isdefined(@__MODULE__, :_gpu_resources_freed)
    const _gpu_resources_freed = Ref(false)
end
if !isdefined(@__MODULE__, :_mlp_cpu_fallback_cache)
    const _mlp_cpu_fallback_cache = Dict{Int, NamedTuple{(:gate, :up, :down), Tuple{Matrix{Float32}, Matrix{Float32}, Matrix{Float32}}}}()
end
if !isdefined(@__MODULE__, :_mlp_cpu_fallback_warned)
    const _mlp_cpu_fallback_warned = Set{Int}()
end

_mlp_cpu_cache() = getfield(@__MODULE__, :_mlp_cpu_fallback_cache)
_mlp_cpu_warned() = getfield(@__MODULE__, :_mlp_cpu_fallback_warned)

function cleanup_all!()
    if _gpu_resources_freed[]
        return nothing
    end

    _try_free_gpu!(x) = try
        x === nothing || oneAPI.unsafe_free!(x)
    catch
        if hasproperty(x, :data)
            try
                oneAPI.unsafe_free!(getproperty(x, :data))
            catch
            end
        end
    end
    _try_clear_cpu!(x) = try
        x === nothing && return nothing
        if x isa AbstractVector
            fill!(x, zero(eltype(x)))
            resize!(x, 0)
        elseif x isa AbstractArray
            fill!(x, zero(eltype(x)))
        end
    catch
    end

    try
        oneAPI.synchronize()
    catch
    end

    _free_binding_if_defined!(name::Symbol) = begin
        if isdefined(@__MODULE__, name)
            _try_free_gpu!(getfield(@__MODULE__, name))
        end
        nothing
    end
    _clear_binding_if_defined!(name::Symbol) = begin
        if isdefined(@__MODULE__, name)
            _try_clear_cpu!(getfield(@__MODULE__, name))
        end
        nothing
    end
    if isdefined(@__MODULE__, :_mlp_cpu_fallback_cache)
        empty!(getfield(@__MODULE__, :_mlp_cpu_fallback_cache))
    end
    if isdefined(@__MODULE__, :_mlp_cpu_fallback_warned)
        empty!(getfield(@__MODULE__, :_mlp_cpu_fallback_warned))
    end
    blocks = isdefined(@__MODULE__, :loaded_blocks) ? getfield(@__MODULE__, :loaded_blocks) : ()
    ssm_cache_list = isdefined(@__MODULE__, :ssm_caches) ? getfield(@__MODULE__, :ssm_caches) : ()
    kv_cache_list = isdefined(@__MODULE__, :kv_caches) ? getfield(@__MODULE__, :kv_caches) : ()

    # 1. Free blocks
    for blk in blocks
        _try_free_gpu!(blk.attn_norm.weight)
        _try_free_gpu!(blk.post_attn_norm.weight)
        if blk.is_ssm
            _try_free_gpu!(blk.attn_qkv_weight)
            _try_free_gpu!(blk.attn_gate_weight)
            _try_free_gpu!(blk.ssm_out_weight)
            _try_free_gpu!(blk.ssm_conv1d_weight)
            _try_free_gpu!(blk.ssm_alpha_weight)
            _try_free_gpu!(blk.ssm_beta_weight)
            _try_free_gpu!(blk.ssm_a)
            _try_free_gpu!(blk.ssm_dt_bias)
            _try_free_gpu!(blk.ssm_norm.weight)
            _try_clear_cpu!(blk.ssm_alpha_weight_cpu)
            _try_clear_cpu!(blk.ssm_beta_weight_cpu)
            _try_clear_cpu!(blk.ssm_a_cpu)
            _try_clear_cpu!(blk.ssm_dt_bias_cpu)
        else
            _try_free_gpu!(blk.attn_q_weight)
            _try_free_gpu!(blk.attn_k_weight)
            _try_free_gpu!(blk.attn_v_weight)
            _try_free_gpu!(blk.attn_output_weight)
            _try_free_gpu!(blk.attn_q_norm.weight)
            _try_free_gpu!(blk.attn_k_norm.weight)
        end
        _try_free_gpu!(blk.ffn_gate_weight)
        _try_free_gpu!(blk.ffn_up_weight)
        _try_free_gpu!(blk.ffn_down_weight)
    end

    # 2. Free Global weights
    _free_binding_if_defined!(:lm_head_weight)
    _free_binding_if_defined!(:logits_work_buf)
    if isdefined(@__MODULE__, :final_rms_pre)
        _try_free_gpu!(getfield(@__MODULE__, :final_rms_pre).weight)
    end
    _free_binding_if_defined!(:rope_sin_cache_gpu)
    _free_binding_if_defined!(:rope_cos_cache_gpu)
    _free_binding_if_defined!(:embed_gpu)
    _free_binding_if_defined!(:hidden_work_buf)
    _free_binding_if_defined!(:hidden_out_buf)
    _clear_binding_if_defined!(:embed)
    _clear_binding_if_defined!(:logits_host)
    _clear_binding_if_defined!(:rope_sin_cache)
    _clear_binding_if_defined!(:rope_cos_cache)

    # 3. Free SSM Caches
    for cache in ssm_cache_list
        _try_free_gpu!(cache.conv)
        _try_free_gpu!(cache.qkv_proj)
        _try_free_gpu!(cache.z_buf)
        _try_free_gpu!(cache.x_conv)
        _try_free_gpu!(cache.y_all)
        _try_free_gpu!(cache.branch_out)
        _try_clear_cpu!(cache.h)
        _try_clear_cpu!(cache.x_norm_cpu)
        _try_clear_cpu!(cache.x_norm_cpu32)
        _try_clear_cpu!(cache.x_conv_cpu)
        _try_clear_cpu!(cache.y_all_cpu)
        _try_clear_cpu!(cache.alpha_proj)
        _try_clear_cpu!(cache.beta_proj)
        _try_clear_cpu!(cache.q_norm_buf)
        _try_clear_cpu!(cache.k_norm_buf)
        _try_clear_cpu!(cache.tmp_head)
    end

    # 4. Free KV Caches
    for cache in kv_cache_list
        _try_free_gpu!(cache.k)
        _try_free_gpu!(cache.v)
        _try_free_gpu!(cache.q_all)
        _try_free_gpu!(cache.k_buf)
        _try_free_gpu!(cache.v_buf)
        _try_free_gpu!(cache.scores)
        _try_free_gpu!(cache.attn_out_buf)
        _try_free_gpu!(cache.branch_out)
        _try_free_gpu!(cache.mlp_gate)
        _try_free_gpu!(cache.mlp_up)
        _try_free_gpu!(cache.rope_q_tmp)
        _try_free_gpu!(cache.rope_k_tmp)
        _try_free_gpu!(cache.norm1_buf)
        _try_free_gpu!(cache.norm2_buf)
    end

    _gpu_resources_freed[] = true
    GC.gc(true)
    println("Module-owned GPU and CPU buffers cleared.")
    return nothing
end

free_all_gpu!() = cleanup_all!()

function ensure_runtime_available!()
    if _gpu_resources_freed[]
        error("TestInference runtime buffers were freed by cleanup_all!(). Call reload_globals!() or reload examples/inference.jl before running inference again.")
    end
    return nothing
end

function reload_globals!()
    parent = parentmodule(@__MODULE__)
    module_name = nameof(@__MODULE__)
    file_path = joinpath(@__DIR__, "inference.jl")
    Base.invokelatest(Base.include, parent, file_path)
    return getfield(parent, module_name)
end

function maybe_live_module()
    if !_gpu_resources_freed[]
        return nothing
    end
    parent = parentmodule(@__MODULE__)
    module_name = nameof(@__MODULE__)
    if isdefined(parent, module_name)
        latest = getfield(parent, module_name)
        if latest !== @__MODULE__
            return latest
        end
    end
    return nothing
end

function reset_caches!()
    ensure_runtime_available!()
    for cache in ssm_caches
        fill!(cache.conv, Float16(0.0))
        fill!(cache.h, 0.0f0)
    end
end

function load_hidden!(dest, toks)
    ensure_runtime_available!()
    if hasproperty(toks, :data)
        copyto!(dest, toks)
    elseif toks isa AbstractArray{Float16}
        copyto!(dest, toks)
    else
        copyto!(dest, Float16.(toks))
    end
    return dest
end
