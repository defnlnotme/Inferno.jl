# Trace/verify/bisect helpers for inference
using LinearAlgebra: dot, norm, sqrt

function _array_stats(x)
    x_cpu = x isa AbstractArray ? Array(x) : collect(x)
    vals = Float32.(vec(x_cpu))
    isempty(vals) && return (; len=0, finite_count=0, min=NaN32, max=NaN32, mean_abs=NaN32, rms=NaN32)
    finite_mask = isfinite.(vals)
    finite_vals = vals[finite_mask]
    if isempty(finite_vals)
        return (; len=length(vals), finite_count=0, min=NaN32, max=NaN32, mean_abs=NaN32, rms=NaN32)
    end
    rms = sqrt(sum(abs2, finite_vals) / length(finite_vals))
    mean_abs = sum(abs, finite_vals) / length(finite_vals)
    return (; len=length(vals), finite_count=length(finite_vals), min=minimum(finite_vals), max=maximum(finite_vals), mean_abs=mean_abs, rms=rms)
end

function _topk_pairs(logits; k::Int=10)
    logits_cpu = Float32.(Array(logits))
    kk = min(k, length(logits_cpu))
    perm = partialsortperm(logits_cpu, 1:kk; rev=true)
    return [(id=i, logit=logits_cpu[i], text=Inferno.Tokenizer.decode(tokenizer, [i])) for i in perm]
end

function trace_forward_token!(tok_id::Integer, pos::Integer; topk::Int=10, run_final_norm::Bool=true)
    ensure_runtime_available!()
    @views copyto!(hidden_work_buf, embed_gpu[tok_id, :])
    x = hidden_work_buf
    token_trace = Vector{Any}(undef, config.num_hidden_layers)

    if pos == 1
        reset_caches!()
    end

    for i in 1:config.num_hidden_layers
        blk = loaded_blocks[i]
        cache = kv_caches[i]

        x_norm1 = cache.norm1_buf
        Model.rmsnorm!(x_norm1, x, blk.attn_norm)
        branch1 = blk.is_ssm ?
            process_branch!(Val(:ssm), blk, x_norm1, ssm_caches[i], config) :
            process_branch!(Val(:attn), blk, x_norm1, cache, config, pos)

        branch1_stats = _array_stats(branch1)
        x .+= branch1
        after_attn_stats = _array_stats(x)

        x_norm2 = cache.norm2_buf
        Model.rmsnorm!(x_norm2, x, blk.post_attn_norm)
        mlp_out = process_mlp!(blk, x_norm2, cache)
        mlp_stats = _array_stats(mlp_out)
        x .+= mlp_out
        after_mlp_stats = _array_stats(x)

        token_trace[i] = (
            layer=i,
            kind=blk.is_ssm ? :ssm : :attn,
            norm1=_array_stats(x_norm1),
            branch1=branch1_stats,
            after_branch1=after_attn_stats,
            norm2=_array_stats(x_norm2),
            mlp=mlp_stats,
            after_mlp=after_mlp_stats,
        )
    end

    hidden = run_final_norm ? Model.rmsnorm!(hidden_out_buf, x, final_rms_pre) : x
    logits = project_to_logits!(logits_work_buf, hidden)
    return (; token_id=tok_id, position=pos, hidden_stats=_array_stats(hidden), logits_stats=_array_stats(logits), topk=_topk_pairs(logits; k=topk), layers=token_trace)
end

function verify_rope_math(; pos::Int=1, atol::Real=1e-3)
    n_heads_q = config.num_attention_heads
    n_heads_kv = config.num_key_value_heads
    q_cpu = rand(Float16, config.head_dim, n_heads_q)
    k_cpu = rand(Float16, config.head_dim, n_heads_kv)
    q_gpu = oneArray(copy(q_cpu))
    k_gpu = oneArray(copy(k_cpu))
    cache = KVCache(config)
    apply_rope!(q_gpu, k_gpu, pos, config, cache)
    apply_rope_reference_cpu!(q_cpu, k_cpu, pos, config)
    q_out = Array(q_gpu)
    k_out = Array(k_gpu)
    q_err = maximum(abs.(Float32.(q_out) .- Float32.(q_cpu)))
    k_err = maximum(abs.(Float32.(k_out) .- Float32.(k_cpu)))
    ok = (q_err <= atol) && (k_err <= atol)
    return (; ok, q_err, k_err, atol)
end

function verify_ssm_math(; layer_index::Union{Nothing,Int}=nothing, atol::Real=5e-3)
    idx = layer_index
    if idx === nothing
        idx = findfirst(b -> b.is_ssm, loaded_blocks)
        idx === nothing && error("No SSM layer found in loaded_blocks")
    end
    block = loaded_blocks[idx]
    block.is_ssm || error("Layer $idx is not an SSM layer")

    x = oneArray(rand(Float16, config.hidden_size))
    x_norm = block.attn_norm(x)
    ref_cache = SSMState(config)
    opt_cache = SSMState(config)

    ref_out = process_branch_ssm_reference!(block, x_norm, ref_cache, config)
    opt_out = process_branch!(Val(:ssm), block, x_norm, opt_cache, config)

    ref_out_cpu = Array(ref_out)
    opt_out_cpu = Array(opt_out)
    out_err = maximum(abs.(Float32.(ref_out_cpu) .- Float32.(opt_out_cpu)))
    state_err = maximum(abs.(ref_cache.h .- opt_cache.h))
    ok = (out_err <= atol) && (state_err <= atol)
    return (; ok, layer_index=idx, out_err, state_err, atol)
end

function verify_no_math_regressions(; pos::Int=1, rope_atol::Real=1e-3, ssm_atol::Real=5e-3)
    rope = verify_rope_math(; pos=pos, atol=rope_atol)
    ssm = verify_ssm_math(; atol=ssm_atol)
    ok = rope.ok && ssm.ok
    return (; ok, rope, ssm)
end

function bisect_prompt(
    prompt::AbstractString=prompt;
    token_ids::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
    topk::Int=10,
    stop_on_first_anomaly::Bool=false,
    rms_jump_threshold::Real=8,
)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(
            getfield(latest, :bisect_prompt),
            prompt;
            token_ids=token_ids,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            topk=topk,
            stop_on_first_anomaly=stop_on_first_anomaly,
            rms_jump_threshold=rms_jump_threshold,
        )
    end
    ids = token_ids === nothing ?
        encode_prompt(prompt; add_bos=false, use_chat_template=use_chat_template, system_prompt=system_prompt, enable_thinking=enable_thinking) :
        Int.(collect(token_ids))
    isempty(ids) && error("Prompt encoded to zero tokens")

    token_traces = Any[]
    anomalies = Any[]

    for (pos, tok_id) in enumerate(ids)
        token_trace = trace_forward_token!(tok_id, pos; topk=topk, run_final_norm=pos == length(ids))
        push!(token_traces, token_trace)

        for layer_info in token_trace.layers
            branch_rms = layer_info.branch1.rms
            after_rms = layer_info.after_branch1.rms
            mlp_rms = layer_info.mlp.rms
            final_rms = layer_info.after_mlp.rms

            if layer_info.after_mlp.finite_count != layer_info.after_mlp.len
                push!(anomalies, (; position=pos, layer=layer_info.layer, reason=:non_finite_hidden, stats=layer_info.after_mlp))
            elseif isfinite(branch_rms) && isfinite(after_rms) && branch_rms > 0 && after_rms / branch_rms > rms_jump_threshold
                push!(anomalies, (; position=pos, layer=layer_info.layer, reason=:branch_rms_jump, branch_rms, after_rms))
            elseif isfinite(mlp_rms) && isfinite(final_rms) && mlp_rms > 0 && final_rms / mlp_rms > rms_jump_threshold
                push!(anomalies, (; position=pos, layer=layer_info.layer, reason=:mlp_rms_jump, mlp_rms, final_rms))
            end

            if stop_on_first_anomaly && !isempty(anomalies)
                break
            end
        end

        if stop_on_first_anomaly && !isempty(anomalies)
            break
        end
    end

    final_trace = token_traces[end]
    return (
        prompt=String(prompt),
        prompt_token_ids=ids,
        prompt_tokens=[Inferno.Tokenizer.decode(tokenizer, [id]) for id in ids],
        final_hidden_stats=final_trace.hidden_stats,
        final_logits_stats=final_trace.logits_stats,
        final_topk=final_trace.topk,
        token_traces=token_traces,
        anomalies=anomalies,
    )
end

function print_bisect_report(
    prompt::AbstractString=prompt;
    token_ids::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    use_chat_template::Bool=true,
    system_prompt::Union{Nothing, AbstractString}=nothing,
    enable_thinking::Bool=false,
    topk::Int=10,
    io::IO=stdout,
)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(
            getfield(latest, :print_bisect_report),
            prompt;
            token_ids=token_ids,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            topk=topk,
            io=io,
        )
    end
    report = bisect_prompt(
        prompt;
        token_ids=token_ids,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        topk=topk,
    )

    println(io, "Prompt token ids: ", report.prompt_token_ids)
    println(io, "Prompt tokens: ", report.prompt_tokens)
    println(io, "Final hidden stats: ", report.final_hidden_stats)
    println(io, "Final logits stats: ", report.final_logits_stats)
    println(io, "Final top-$topk logits:")
    for item in report.final_topk
        println(io, "  ", item.id, " | ", repr(item.text), " | ", item.logit)
    end

    if isempty(report.anomalies)
        println(io, "No numeric anomalies detected.")
    else
        println(io, "Anomalies:")
        for item in report.anomalies
            println(io, "  ", item)
        end
    end

    println(io, "Per-token / per-layer RMS summary:")
    for token_trace in report.token_traces
        println(io, "  pos=", token_trace.position, " tok=", token_trace.token_id, " ", repr(Inferno.Tokenizer.decode(tokenizer, [token_trace.token_id])))
        for layer in token_trace.layers
            println(
                io,
                "    layer=", layer.layer,
                " kind=", layer.kind,
                " norm1=", layer.norm1.rms,
                " branch=", layer.branch1.rms,
                " after1=", layer.after_branch1.rms,
                " norm2=", layer.norm2.rms,
                " mlp=", layer.mlp.rms,
                " after2=", layer.after_mlp.rms,
            )
        end
    end

    return report
end

function dump_token_embedding(tok_id::Integer; topn::Int=16)
    ensure_runtime_available!()
    emb_cpu = Float32.(Array(@view embed_gpu[tok_id, :]))
    n = min(topn, length(emb_cpu))
    return (
        token_id=tok_id,
        token=Inferno.Tokenizer.decode(tokenizer, [tok_id]),
        stats=_array_stats(emb_cpu),
        head=emb_cpu[1:n],
    )
end

function dump_prelogit_hidden(tok_id::Integer; pos::Integer=1, topn::Int=16)
    ensure_runtime_available!()
    @views copyto!(hidden_work_buf, embed_gpu[tok_id, :])
    x = hidden_work_buf
    forward_hidden_core!(x, loaded_blocks, config, file, pos)
    hidden = Model.rmsnorm!(hidden_out_buf, x, final_rms_pre)
    hidden_cpu = Float32.(Array(hidden))
    n = min(topn, length(hidden_cpu))
    return (
        token_id=tok_id,
        token=Inferno.Tokenizer.decode(tokenizer, [tok_id]),
        residual_stats=_array_stats(x),
        hidden_stats=_array_stats(hidden),
        residual_head=Float32.(Array(x))[1:n],
        hidden_head=hidden_cpu[1:n],
    )
end

function trace_single_token_logits(tok_id::Integer; pos::Integer=1, topk::Int=20)
    ensure_runtime_available!()
    @views copyto!(hidden_work_buf, embed_gpu[tok_id, :])
    x = hidden_work_buf
    forward_hidden_core!(x, loaded_blocks, config, file, pos)
    hidden = Model.rmsnorm!(hidden_out_buf, x, final_rms_pre)
    logits = project_to_logits!(logits_work_buf, hidden)
    return (
        token_id=tok_id,
        token=Inferno.Tokenizer.decode(tokenizer, [tok_id]),
        embedding=dump_token_embedding(tok_id),
        residual_stats=_array_stats(x),
        hidden_stats=_array_stats(hidden),
        logits_stats=_array_stats(logits),
        topk=_topk_pairs(logits; k=topk),
    )
end

function print_single_token_report(tok_id::Integer; pos::Integer=1, topk::Int=20, io::IO=stdout)
    if (latest = maybe_live_module()) !== nothing
        return Base.invokelatest(getfield(latest, :print_single_token_report), tok_id; pos=pos, topk=topk, io=io)
    end
    report = trace_single_token_logits(tok_id; pos=pos, topk=topk)
    println(io, "Token id: ", report.token_id, " ", repr(report.token))
    println(io, "Embedding stats: ", report.embedding.stats)
    println(io, "Residual stats before final norm: ", report.residual_stats)
    println(io, "Hidden stats after final norm: ", report.hidden_stats)
    println(io, "Logits stats: ", report.logits_stats)
    println(io, "Top-$topk logits:")
    for item in report.topk
        println(io, "  ", item.id, " | ", repr(item.text), " | ", item.logit)
    end
    return report
end
