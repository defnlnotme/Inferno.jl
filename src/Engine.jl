module Engine

using ..Model
using ..Tokenizer
using ..oneAPI
using Random

export generate, generate_stream, sample, stream_to_stdout

function simple_sample(probs::Vector{Float16})
    r = rand()
    cum = Float16(0.0)
    for (i, p) in enumerate(probs)
        cum += p
        if r <= cum
            return i
        end
    end
    return length(probs)
end

function sample(logits::Vector{Float16}, temperature::Float16, top_p::Float16, top_k::Int=40)
    return cpu_sample(logits, temperature, top_p, top_k)
end

# CPU sampling implementation (original code)
function cpu_sample(logits::Vector{Float16}, temperature::Float16, top_p::Float16, top_k::Int=40)
    # Sanitize logits - replace NaN and Inf with -Inf so they're never selected
    @inbounds for i in eachindex(logits)
        if !isfinite(logits[i])
            logits[i] = -Float16(Inf)
        end
    end

    if temperature == Float16(0.0)
        return argmax(logits)
    end

    # Top-k sampling
    if top_k > 0 && top_k < length(logits)
        # Get top-k indices
        top_k_indices = partialsortperm(logits, 1:top_k, rev=true)

        # Create a mask for non-top-k elements
        for i in eachindex(logits)
            if i ∉ top_k_indices
                logits[i] = -Float16(Inf)
            end
        end
    end

    # Temperature scaling + softmax in one pass (minimise allocations)
    inv_temp = Float16(1.0) / temperature
    mx = -Float16(Inf)
    @inbounds for v in logits
        scaled = v * inv_temp
        mx = scaled > mx ? scaled : mx
    end
    probs = similar(logits)
    s = Float16(0.0)
    @inbounds for i in eachindex(logits)
        e = exp(logits[i] * inv_temp - mx)
        probs[i] = e
        s += e
    end
    inv_s = Float16(1.0) / s
    @inbounds probs .*= inv_s

    # Top-p (Nucleus) - partialsortperm avoids full O(N log N) sort over 150k vocab
    if top_p < Float16(1.0)
        n = length(probs)
        k_max = min(n, 1024) # top-p usually keeps well under 1024 tokens
        top_indices = partialsortperm(probs, 1:k_max, rev=true)
        cum = Float16(0.0)
        cutoff = k_max
        for (rank, idx) in enumerate(top_indices)
            cum += probs[idx]
            if cum >= top_p
                cutoff = rank
                break
            end
        end
        # Re-build probs with only the kept tokens (renormalized)
        fill!(probs, Float16(0.0))
        renorm = Float16(0.0)
        @inbounds for rank in 1:cutoff
            idx = top_indices[rank]
            e = exp(logits[idx] * inv_temp - mx)
            probs[idx] = e
            renorm += e
        end
        inv_renorm = Float16(1.0) / renorm
        @inbounds for rank in 1:cutoff
            probs[top_indices[rank]] *= inv_renorm
        end
    end

    return simple_sample(probs)
end

# CPU sampling from already scaled logits
function cpu_sample_from_scaled(scaled_logits::Vector{Float16}, temperature::Float16, top_p::Float16, top_k::Int=40)
    mx = maximum(scaled_logits)
    exp_vals = exp.(scaled_logits .- mx)
    probs = exp_vals ./ sum(exp_vals)

    # Top-k sampling
    if top_k > 0 && top_k < length(scaled_logits)
        # Get top-k indices based on probabilities
        top_k_indices = partialsortperm(probs, 1:min(top_k, length(probs)), rev=true)

        # Zero out non-top-k probabilities
        for i in eachindex(probs)
            if i ∉ top_k_indices
                probs[i] = Float16(0.0)
            end
        end

        # Renormalize
        total_prob = sum(probs)
        if total_prob > Float16(0.0)
            probs ./= total_prob
        end
    end

    # Top-p sampling
    if top_p < Float16(1.0)
        n = length(probs)
        k_max = min(n, 1024)
        top_indices = partialsortperm(probs, 1:k_max, rev=true)
        cum = Float16(0.0)
        cutoff = k_max
        for (rank, idx) in enumerate(top_indices)
            cum += probs[idx]
            if cum >= top_p
                cutoff = rank
                break
            end
        end

        # Rebuild probs with top-p tokens
        fill!(probs, Float16(0.0))
        for rank in 1:cutoff
            idx = top_indices[rank]
            probs[idx] = exp_vals[idx]
        end
        probs ./= sum(probs)
    end

    return simple_sample(probs)
end

# mask_and_sample: wrapper over `sample` which will avoid returning PAD-like tokens
# when streaming. We defensively re-sample a few times while masking offending ids.
# This is intentionally conservative (max_attempts) to avoid infinite loops.
function mask_and_sample(logits::AbstractVector{Float16}, pad_ids::Vector{Int}, temperature::Float16, top_p::Float16, top_k::Int=40; max_attempts::Int=5)
    # Reuse the vector in-place if already CPU, otherwise collect once
    logits_copy = logits isa Vector{Float16} ? copy(logits) : Vector{Float16}(logits)

    # Pre-mask known PAD-like token ids
    if !isempty(pad_ids)
        @inbounds for pid in pad_ids
            if 1 <= pid <= length(logits_copy)
                logits_copy[pid] = -Float16(Inf)
            end
        end
    end

    for attempt in 1:max_attempts
        cand = sample(logits_copy, temperature, top_p, top_k)
        if !isempty(pad_ids) && cand <= length(logits_copy) && logits_copy[cand] <= -1e8
            logits_copy[cand] = -Float16(Inf)
            continue
        end
        return cand
    end

    return argmax(logits_copy)
end

function generate_stream(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens=100, temperature=Float16(0.7), top_p=Float16(0.8), top_k=20, stop_token=nothing)

    tokens = Tokenizer.encode(tok, prompt)
    if isempty(tokens)
        return Channel{String}(0) do chan
            close(chan)
        end
    end

    kv_caches = Vector{Model.KVCache}()
    return Channel{String}(32) do chan
        try
            # Ensure we're on the correct GPU device
            devs = collect(oneAPI.devices())
            if !isempty(devs)
                first_weight = model.layers[1].mlp.gate_weight
                weight_device = device(first_weight)
                oneAPI.device!(weight_device)
            end

            Model.reset_states!(model)
            kv_caches = [Model.init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

            # Prefill
            logits = Model.forward!(model, tokens, 0, kv_caches)
            curr_pos = length(tokens)

            # Sample from last position
            logits_vec = vec(collect(logits[:, end]))
            last_token = mask_and_sample(logits_vec, [tok.eos_id], Float16(temperature), Float16(top_p), top_k)

            token_str = Tokenizer.decode(tok, [last_token])
            put!(chan, token_str)

            # Decode loop
            for step in 1:(max_tokens-1)
                if last_token == (stop_token === nothing ? tok.eos_id : stop_token)
                    break
                end

                logits = Model.forward!(model, [last_token], curr_pos, kv_caches)
                curr_pos += 1

                logits_vec = vec(collect(logits[:, 1]))
                last_token = mask_and_sample(logits_vec, [tok.eos_id], Float16(temperature), Float16(top_p), top_k)

                token_str = Tokenizer.decode(tok, [last_token])
                put!(chan, token_str)

                if step % 4 == 0
                    GC.gc(false)
                end
            end

            oneAPI.synchronize()

        catch e
            if !(e isa InterruptException) && !(e isa InvalidStateException)
                @error "ERROR during generation stream" exception = (e, catch_backtrace())
            end
        finally
            try
                if !isempty(kv_caches)
                    Model.free_all_kv_caches!(kv_caches)
                end
            catch
            end
            try
                GC.gc(true)
            catch
            end
            try
                close(chan)
            catch
            end
        end
    end
end

function generate(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens=100, temperature=Float16(0.7), top_p=Float16(0.8), top_k=20, stop_token=nothing)

    stream = generate_stream(model, tok, prompt; max_tokens, temperature, top_p, top_k, stop_token)
    res = String[]
    for token_str in stream
        push!(res, token_str)
    end
    return join(res)
end

# Stream directly to stdout for interactive use
function stream_to_stdout(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens=100, temperature=Float16(0.7), top_p=Float16(0.8), top_k=20)
    stream = generate_stream(model, tok, prompt; max_tokens, temperature, top_p, top_k)
    for token_str in stream
        print(token_str)
    end
    println()
end

end # module
