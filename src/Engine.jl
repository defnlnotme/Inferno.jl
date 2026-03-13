module Engine

using ..Model
using ..Tokenizer
using Random

export generate, generate_stream, sample

function simple_sample(probs::Vector{Float32})
    r = rand()
    cum = 0.0f0
    for (i, p) in enumerate(probs)
        cum += p
        if r <= cum
            return i
        end
    end
    return length(probs)
end

function sample(logits::Vector{Float32}, temperature::Float32, top_p::Float32)
    # Sanitize logits
    logits[.!isfinite.(logits)] .= -1e9f0

    if temperature == 0.0f0
        return argmax(logits)
    end

    # Temperature
    logits32 = logits ./ temperature

    # Softmax
    exp_logits = exp.(logits32 .- maximum(logits32))
    probs = exp_logits ./ sum(exp_logits)

    # Top-p (Nucleus)
    p_indices = sortperm(probs, rev=true)
    p_sort = probs[p_indices]
    p_cum = cumsum(p_sort)

    cutoff_idx = findfirst(x -> x >= top_p, p_cum)
    if cutoff_idx !== nothing
        # Zero out tokens below cutoff
        for i in (cutoff_idx+1):length(p_indices)
            probs[p_indices[i]] = 0.0f0
        end
        probs ./= sum(probs)
    end

    return simple_sample(probs)
end

# mask_and_sample: wrapper over `sample` which will avoid returning PAD-like tokens
# when streaming. We defensively re-sample a few times while masking offending ids.
# This is intentionally conservative (max_attempts) to avoid infinite loops.
function mask_and_sample(logits::Vector{Float32}, pad_ids::Vector{Int}, temperature::Float32, top_p::Float32; max_attempts::Int=5)
    # Work on a copy since `sample` mutates its input
    logits_copy = copy(logits)

    # Pre-mask known PAD-like token ids to avoid ever selecting them.
    # This avoids expensive decode calls per candidate and is deterministic.
    pad_mask = falses(length(logits_copy))
    if !isempty(pad_ids)
        for pid in pad_ids
            if 1 <= pid <= length(logits_copy)
                pad_mask[pid] = true
                logits_copy[pid] = -1e9f0
            end
        end
    end

    for attempt in 1:max_attempts
        cand = sample(logits_copy, temperature, top_p)
        # If candidate is in the PAD mask, ban it for further attempts and retry.
        if cand <= length(pad_mask) && pad_mask[cand]
            logits_copy[cand] = -1e9f0
            continue
        end
        return cand
    end

    # If attempts exhausted, fall back to the raw sample on original logits.
    # This is conservative: we try to avoid returning PAD-like tokens but won't
    # return an empty result if masking removes all candidates.
    return sample(logits, temperature, top_p)
end

function generate_stream(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens=64, temperature=0.7f0, top_p=0.9f0, stop_token=nothing)

    tokens = Tokenizer.encode(tok, prompt)
    if isempty(tokens)
        return Channel{String}(0) do chan
            close(chan)
        end
    end

    return Channel{String}(32) do chan
        try
            Model.reset_states!(model)
            # Initialize KV Caches (pos tracked inside cache)
            kv_caches = [Model.init_kv_cache(
                model.config.head_dim,
                model.config.num_key_value_heads,
                model.config.max_position_embeddings)
                for _ in 1:model.config.num_hidden_layers]

            # Prefill: pass pos=0, the KV caches will accumulate internally
            logits = Model.forward!(model, tokens, 0, kv_caches)
            last_token = mask_and_sample(
                vec(logits[:, end]), [tok.eos_id], Float32(temperature), Float32(top_p))

            token_str = Tokenizer.decode(tok, [last_token])
            put!(chan, token_str)

            # Decode: each step passes current cache.pos as the position
            for _ in 1:(max_tokens-1)
                if last_token == (stop_token === nothing ? tok.eos_id : stop_token)
                    break
                end

                curr_pos = kv_caches[1].pos  # all caches share same length after prefill
                logits = Model.forward!(model, [last_token], curr_pos, kv_caches)
                last_token = mask_and_sample(
                    vec(logits[:, 1]), [tok.eos_id], Float32(temperature), Float32(top_p))

                token_str = Tokenizer.decode(tok, [last_token])
                put!(chan, token_str)
            end
        catch e
            @error "ERROR during generation stream" exception = (e, catch_backtrace())
        finally
            close(chan)
        end
    end
end

function generate(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens=64, temperature=0.7f0, top_p=0.9f0, stop_token=nothing)

    stream = generate_stream(model, tok, prompt; max_tokens, temperature, top_p, stop_token)
    res = String[]
    for token_str in stream
        push!(res, token_str)
    end
    return join(res)
end

end # module
