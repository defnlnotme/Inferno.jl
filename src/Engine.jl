module Engine

using oneAPI
using ..Model
using ..Tokenizer

export generate, sample_token

"""
    sample_token(logits; temperature=0.7, top_p=0.9) -> Int

Sample a token ID from a logits vector using temperature scaling and nucleus (top-p) sampling.
"""
function sample_token(logits::Vector{Float32}; temperature::Float32=0.7f0, top_p::Float32=0.9f0)
    if temperature <= 0.0f0
        return argmax(logits)
    end

    # Temperature scaling
    scaled = logits ./ temperature

    # Softmax
    m = maximum(scaled)
    probs = exp.(scaled .- m)
    probs ./= sum(probs)

    # Top-p (nucleus) sampling
    sorted_idx = sortperm(probs, rev=true)
    cumsum_probs = cumsum(probs[sorted_idx])
    cutoff = findfirst(>=(top_p), cumsum_probs)
    cutoff = isnothing(cutoff) ? length(probs) : cutoff
    
    # Zero out tokens below the cutoff
    mask = falses(length(probs))
    mask[sorted_idx[1:cutoff]] .= true
    probs[.!mask] .= 0.0f0
    probs ./= sum(probs)

    # Weighted random sample
    r = rand(Float32)
    cumul = 0.0f0
    for i in eachindex(probs)
        cumul += probs[i]
        if r <= cumul
            return i
        end
    end
    return argmax(probs)
end

"""
    generate(model, tokenizer, prompt; max_tokens=128, temperature=0.7, top_p=0.9) -> String

Autoregressive text generation with KV caching.
"""
function generate(model::QwenModel, tok::BPETokenizer, prompt::String;
                  max_tokens::Int=128, temperature::Float32=0.7f0,
                  top_p::Float32=0.9f0, stop_token::Union{Int,Nothing}=nothing)
    
    eos = isnothing(stop_token) ? tok.eos_id : stop_token
    input_ids = Tokenizer.encode(tok, prompt)
    
    # Initialize KV caches for each layer
    config = model.config
    caches = [Model.init_kv_cache(config.head_dim, config.num_key_value_heads, config.max_position_embeddings) 
              for _ in 1:config.num_hidden_layers]
    
    generated_ids = Int[]
    pos = 0

    # Prefill: process entire prompt at once
    # forward! expects Vector{Int}, pos::Int, caches::Vector{KVCache}
    logits = Model.forward!(model, input_ids, pos, caches)
    pos += length(input_ids)
    
    # logits is (vocab, seq)
    # Sample from last position
    last_logits = logits[:, end]
    next_token = sample_token(last_logits; temperature, top_p)
    push!(generated_ids, next_token)

    if next_token == eos
        return Tokenizer.decode(tok, generated_ids)
    end

    # Decode: one token at a time
    for _ in 2:max_tokens
        logits = Model.forward!(model, [next_token], pos, caches)
        pos += 1
        
        last_logits = logits[:, 1]
        next_token = sample_token(last_logits; temperature, top_p)
        push!(generated_ids, next_token)
        
        if next_token == eos
            break
        end
    end

    return Tokenizer.decode(tok, generated_ids)
end

end # module
