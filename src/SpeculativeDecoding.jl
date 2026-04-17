# --- Speculative Decoding Support ---
# This is included inside ModelCPU module

"""
    SpeculativeDecoder

Implements speculative decoding with a draft model for faster generation.
"""
struct SpeculativeDecoder
    draft_model::QwenModelCPU
    target_model::QwenModelCPU
    gamma::Int  # Number of draft tokens to generate per step
end

"""
    sample_from_probs(probs)

Sample a token index from probability distribution.
"""
function sample_from_probs(probs::Vector{Float32})
    cumsum = 0.0f0
    r = rand(Float32)
    for (i, p) in enumerate(probs)
        cumsum += p
        if r <= cumsum
            return i
        end
    end
    return length(probs)
end

"""
    speculate_and_verify(decoder, prompt_tokens, draft_caches, target_caches, position; kwargs...)

Single step of speculative decoding.
"""
function speculate_and_verify(
    decoder::SpeculativeDecoder,
    prompt_tokens::Vector{Int},
    draft_caches::Vector{KVCacheCPU},
    target_caches::Vector{KVCacheCPU},
    position::Int;
    temperature::Float32=1.0f0,
    top_p::Float32=1.0f0,
    top_k::Int=0
)
    # Generate gamma draft tokens
    draft_tokens = Int[]
    draft_logprobs = Float32[]
    
    curr_pos = position
    for _ in 1:decoder.gamma
        input_tokens = isempty(draft_tokens) ? prompt_tokens[end:end] : draft_tokens[end:end]
        
        logits = forward_cpu!(decoder.draft_model, input_tokens, curr_pos, draft_caches)
        logits_vec = vec(logits[:, end])
        
        next_token = softmax_sample(logits_vec; temperature=temperature, top_p=top_p, top_k=top_k)
        
        push!(draft_tokens, next_token)
        logprobs = logits_vec .- maximum(logits_vec)
        logprobs = logprobs .- log(sum(exp.(logprobs)))
        push!(draft_logprobs, logprobs[next_token])
        
        curr_pos += 1
    end
    
    # Target model validation
    full_context = vcat(prompt_tokens[end:end], draft_tokens)
    target_logits = forward_cpu!(decoder.target_model, full_context, position - 1, target_caches)
    
    # Verify tokens
    accepted = Int[]
    
    for i in 1:length(draft_tokens)
        target_logits_i = vec(target_logits[:, i])
        draft_token = draft_tokens[i]
        
        target_logits_i = target_logits_i .- maximum(target_logits_i)
        target_probs = exp.(target_logits_i) ./ sum(exp.(target_logits_i))
        
        draft_p = exp(draft_logprobs[i])
        target_p = target_probs[draft_token]
        
        acceptance_ratio = min(1.0f0, target_p / draft_p)
        
        if rand() < acceptance_ratio
            push!(accepted, draft_token)
        else
            # Reject - sample from residual
            residual = target_probs .- draft_p
            residual = max.(residual, 0.0f0)
            residual = residual ./ sum(residual)
            push!(accepted, sample_from_probs(residual))
            break
        end
    end
    
    # If all gamma accepted, need one more token
    if length(accepted) == decoder.gamma
        target_logits_final = vec(target_logits[:, end])
        final_token = softmax_sample(target_logits_final; temperature=temperature, top_p=top_p, top_k=top_k)
        push!(accepted, final_token)
    end
    
    return accepted, length(accepted), target_logits
end

"""
    generate_speculative_cpu(decoder, prompt_tokens; kwargs...)

Generate tokens using speculative decoding.
"""
function generate_speculative_cpu(
    decoder::SpeculativeDecoder,
    prompt_tokens::Vector{Int};
    max_tokens::Int=512,
    temperature::Float32=1.0f0,
    top_p::Float32=1.0f0,
    top_k::Int=0,
    repetition_penalty::Float32=1.0f0,
    presence_penalty::Float32=0.0f0
)
    # Initialize caches
    draft_caches = [init_kv_cache_cpu(decoder.draft_model.config) for _ in 1:decoder.draft_model.config.num_hidden_layers]
    target_caches = [init_kv_cache_cpu(decoder.target_model.config) for _ in 1:decoder.target_model.config.num_hidden_layers]
    
    reset_states_cpu!(decoder.draft_model)
    reset_states_cpu!(decoder.target_model)
    
    result = copy(prompt_tokens)
    token_counts = Dict{Int,Int}()
    for t in prompt_tokens
        token_counts[t] = get(token_counts, t, 0) + 1
    end
    
    position = length(prompt_tokens)
    tokens_generated = 0
    total_draft_tokens = 0
    total_accepted = 0
    
    while tokens_generated < max_tokens
        accepted, num_accepted, logits = speculate_and_verify(
            decoder, result, draft_caches, target_caches, position;
            temperature=temperature, top_p=top_p, top_k=top_k
        )
        
        for token in accepted
            push!(result, token)
            token_counts[token] = get(token_counts, token, 0) + 1
            position += 1
            tokens_generated += 1
        end
        
        total_draft_tokens += decoder.gamma
        total_accepted += num_accepted
        
        if tokens_generated >= max_tokens
            break
        end
    end
    
    acceptance_rate = total_accepted / total_draft_tokens
    speedup = (tokens_generated - length(prompt_tokens)) / (total_draft_tokens + tokens_generated) * (1 + decoder.gamma)
    
    return result, Dict(
        "acceptance_rate" => acceptance_rate,
        "draft_tokens_generated" => total_draft_tokens,
        "tokens_accepted" => total_accepted,
        "speedup_estimate" => speedup
    )
end
