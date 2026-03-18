# --- GPU Sampling Functions ---

# GPU temperature scaling
function gpu_temperature_scale(logits::oneVector{Float32}, temperature::Float32)
    if temperature == 0.0f0
        return logits  # Return GPU array without CPU transition
    end
    
    inv_temp = 1.0f0 / temperature
    N = length(logits)
    # Optimization: Direct GPU allocation to avoid host-to-device transfer
    scaled_logits = oneArray{Float32}(undef, N)
    fill!(scaled_logits, 0.0f0)
    
    gs = min(N, 256)
    gr = cld(N, gs)
    @oneapi items=gs groups=gr temperature_scale_kernel!(scaled_logits, logits, inv_temp, N)
    
    return scaled_logits
end

# GPU argmax (fallback to CPU for simplicity)
function gpu_argmax(logits::oneVector{Float32})
    logits_cpu = collect(logits)
    return argmax(logits_cpu)
end

module Engine

using ..Model
using ..Tokenizer
using Random
import oneAPI

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

function sample(logits::Vector{Float32}, temperature::Float32, top_p::Float32, top_k::Int=40)
    # Try GPU acceleration if available
    try
        # Move logits to GPU for processing
        logits_gpu = oneArray(logits)
        
        if temperature == 0.0f0
            # Argmax case
            return gpu_argmax(logits_gpu)
        end
        
        # Temperature scaling on GPU
        scaled_logits_gpu = gpu_temperature_scale(logits_gpu, temperature)
        
        # Continue with CPU processing for top-p and sampling
        # (can be GPU-accelerated in future optimizations)
        scaled_logits = collect(scaled_logits_gpu)
        return cpu_sample_from_scaled(scaled_logits, temperature, top_p, top_k)
        
    catch e
        # Fallback to CPU implementation
        return cpu_sample(logits, temperature, top_p, top_k)
    end
end

# CPU sampling implementation (original code)
function cpu_sample(logits::Vector{Float32}, temperature::Float32, top_p::Float32, top_k::Int=40)
    # Sanitize logits
    @inbounds for i in eachindex(logits)
        if !isfinite(logits[i])
            logits[i] = -Inf32
        end
    end

    if temperature == 0.0f0
        return argmax(logits)
    end

    # Top-k sampling
    if top_k > 0 && top_k < length(logits)
        # Get top-k indices
        top_k_indices = partialsortperm(logits, 1:top_k, rev=true)
        
        # Create a mask for non-top-k elements
        for i in eachindex(logits)
            if i ∉ top_k_indices
                logits[i] = -Inf32
            end
        end
    end

    # Temperature scaling + softmax in one pass (minimise allocations)
    inv_temp = 1.0f0 / temperature
    mx = -Inf32
    @inbounds for v in logits
        scaled = v * inv_temp
        mx = scaled > mx ? scaled : mx
    end
    probs = similar(logits)
    s = 0.0f0
    @inbounds for i in eachindex(logits)
        e = exp(logits[i] * inv_temp - mx)
        probs[i] = e
        s += e
    end
    inv_s = 1.0f0 / s
    @inbounds probs .*= inv_s

    # Top-p (Nucleus) — partialsortperm avoids full O(N log N) sort over 150k vocab
    if top_p < 1.0f0
        n = length(probs)
        k_max = min(n, 1024)  # top-p usually keeps well under 1024 tokens
        top_indices = partialsortperm(probs, 1:k_max, rev=true)
        cum = 0.0f0
        cutoff = k_max
        for (rank, idx) in enumerate(top_indices)
            cum += probs[idx]
            if cum >= top_p
                cutoff = rank
                break
            end
        end
        # Re-build probs with only the kept tokens (renormalized)
        fill!(probs, 0.0f0)
        renorm = 0.0f0
        @inbounds for rank in 1:cutoff
            idx = top_indices[rank]
            e = exp(logits[idx] * inv_temp - mx)
            probs[idx] = e
            renorm += e
        end
        inv_renorm = 1.0f0 / renorm
        @inbounds for rank in 1:cutoff
            probs[top_indices[rank]] *= inv_renorm
        end
    end

    return simple_sample(probs)
end

# CPU sampling from already scaled logits
function cpu_sample_from_scaled(scaled_logits::Vector{Float32}, temperature::Float32, top_p::Float32, top_k::Int=40)
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
                probs[i] = 0.0f0
            end
        end
        
        # Renormalize
        total_prob = sum(probs)
        if total_prob > 0.0f0
            probs ./= total_prob
        end
    end
    
    # Top-p sampling
    if top_p < 1.0f0
        n = length(probs)
        k_max = min(n, 1024)
        top_indices = partialsortperm(probs, 1:k_max, rev=true)
        cum = 0.0f0
        cutoff = k_max
        for (rank, idx) in enumerate(top_indices)
            cum += probs[idx]
            if cum >= top_p
                cutoff = rank
                break
            end
        end
        
        # Rebuild probs with top-p tokens
        fill!(probs, 0.0f0)
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
function mask_and_sample(logits::AbstractVector{Float32}, pad_ids::Vector{Int}, temperature::Float32, top_p::Float32, top_k::Int=40; max_attempts::Int=5)
    # Reuse the vector in-place if already CPU, otherwise collect once
    logits_copy = logits isa Vector{Float32} ? copy(logits) : Vector{Float32}(logits)

    # Pre-mask known PAD-like token ids
    if !isempty(pad_ids)
        @inbounds for pid in pad_ids
            if 1 <= pid <= length(logits_copy)
                logits_copy[pid] = -Inf32
            end
        end
    end

    for attempt in 1:max_attempts
        cand = sample(logits_copy, temperature, top_p, top_k)
        if !isempty(pad_ids) && cand <= length(logits_copy) && logits_copy[cand] <= -1e8
            logits_copy[cand] = -Inf32
            continue
        end
        return cand
    end

    return argmax(logits_copy)
end

function generate_stream(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens=64, temperature=0.7f0, top_p=0.8f0, top_k=20, stop_token=nothing)

    tokens = Tokenizer.encode(tok, prompt)
    if isempty(tokens)
        return Channel{String}(0) do chan
            close(chan)
        end
    end

    current_dev = oneAPI.device()
    kv_caches = Vector{Model.KVCache}()  # Initialize empty, populate in try block
    return Channel{String}(32) do chan
        try
            oneAPI.device!(current_dev)
            Model.reset_states!(model)
            # Initialize KV Caches (pos tracked inside cache)
            kv_caches = [Model.init_kv_cache(
                model.config.head_dim,
                model.config.num_key_value_heads,
                model.config.max_position_embeddings)
                for _ in 1:model.config.num_hidden_layers]

            # Prefill: pass pos=0, the KV caches will accumulate internally
            logits = Model.forward!(model, tokens, 0, kv_caches)
            
            # Update curr_pos after prefill
            curr_pos = length(tokens)
            
            # Explicitly collect only the last column of logits for sampling
            logits_vec = vec(collect(logits[:, end]))
            last_token = mask_and_sample(
                logits_vec, [tok.eos_id], Float32(temperature), Float32(top_p), top_k)

            token_str = Tokenizer.decode(tok, [last_token])
            put!(chan, token_str)

            # Decode: each step passes current sequence position
            for step in 1:(max_tokens-1)
                if last_token == (stop_token === nothing ? tok.eos_id : stop_token)
                    break
                end

                logits = Model.forward!(model, [last_token], curr_pos, kv_caches)
                curr_pos += 1
                
                # Explicitly collect the first column (seq=1)
                logits_vec = vec(collect(logits[:, 1]))
                last_token = mask_and_sample(
                    logits_vec, [tok.eos_id], Float32(temperature), Float32(top_p), top_k)

                token_str = Tokenizer.decode(tok, [last_token])
                put!(chan, token_str)
                
                # Force GC every few tokens to prevent memory buildup
                if step % 4 == 0
                    GC.gc(false)
                end
            end
            
            # Synchronize after finishing the stream
            oneAPI.synchronize()
        catch e
            if e isa InterruptException
                # Interrupt - just close the channel, don't rethrow
            elseif e isa InvalidStateException
                # Channel already closed - normal termination
            else
                @error "ERROR during generation stream" exception = (e, catch_backtrace())
            end
        finally
            # Ensure KV caches are freed and memory is released
            try
                if !isempty(kv_caches)
                    Model.free_all_kv_caches!(kv_caches)
                end
            catch
                # Ignore cleanup errors
            end
            
            # Force garbage collection to release VRAM
            try
                GC.gc(true)  # Full GC to ensure VRAM cleanup
            catch
                # Ignore GC errors
            end
            
            # Synchronize GPU to ensure all operations complete
            try
                oneAPI.synchronize()
            catch
                # Ignore sync errors
            end
            
            # Ensure channel is closed even on error
            try
                close(chan)
            catch
                # Channel may already be closed
            end
        end
    end
end

function generate(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens=64, temperature=0.7f0, top_p=0.8f0, top_k=20, stop_token=nothing)

    stream = generate_stream(model, tok, prompt; max_tokens, temperature, top_p, top_k, stop_token)
    res = String[]
    for token_str in stream
        push!(res, token_str)
    end
    return join(res)
end

end # module
