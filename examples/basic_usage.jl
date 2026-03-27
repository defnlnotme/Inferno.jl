# Basic usage example for Inferno.jl CPU backend
# Run with: julia --project=. examples/basic_usage.jl

using Inferno

# Path to your GGUF model
const MODEL_PATH = get(ENV, "INFERNO_MODEL_PATH", "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("Loading model from: $MODEL_PATH")
model, file = LoaderCPU.load_model_cpu(MODEL_PATH)
println("Model loaded successfully!")
println("  Hidden size: $(model.config.hidden_size)")
println("  Layers: $(model.config.num_hidden_layers)")
println("  Vocab size: $(model.config.vocab_size)")

# Get tokenizer data
tokens_data = file.metadata["tokenizer.ggml.tokens"]

# Simple tokenization function (production use requires proper BPE tokenizer)
function simple_tokenize(text::String, tokens_data)
    # This is a hack - for production use the Tokenizer module
    # For now, just find whole-word matches
    token_ids = Int[]
    remaining = text
    while !isempty(remaining)
        found = false
        # Try longest match first
        for len in length(remaining):-1:1
            candidate = SubString(remaining, 1, len)
            for (i, t) in enumerate(tokens_data)
                # Handle Ġ prefix (space marker in GPT-2 style tokenizers)
                clean_t = replace(t, "Ġ" => " ")
                if clean_t == candidate || t == candidate
                    push!(token_ids, i - 1)  # 0-indexed
                    remaining = length(candidate) < length(remaining) ? SubString(remaining, len + 1) : ""
                    found = true
                    break
                end
            end
            found && break
        end
        if !found
            # Skip unknown character
            remaining = length(remaining) > 1 ? SubString(remaining, 2) : ""
        end
    end
    return token_ids
end

# Decode function
function decode_tokens(token_ids::Vector{Int}, tokens_data)
    join([replace(tokens_data[t + 1], "Ġ" => " ") for t in token_ids])
end

# Example 1: Manual token-by-token generation
println("\n" * "="^50)
println("Example 1: Manual Generation")
println("="^50)

# Use some known tokens
# "The" is typically token 562 (with Ġ prefix = " The")
# Try a simple prompt
prompt_tokens = [562]  # " The"

println("Starting with token 562 (\" The\")")

# Initialize caches and states
caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
ModelCPU.reset_states_cpu!(model)

tokens = copy(prompt_tokens)
for i in 1:10
    tok = tokens[end]
    pos = length(tokens) - 1
    
    x = view(model.embed, :, tok)
    for (j, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, caches[j])
    end
    x = model.final_norm(x)
    logits = model.lm_head * x
    
    next_token = argmax(logits)
    push!(tokens, next_token)
end

output_text = decode_tokens(tokens, tokens_data)
println("\nGenerated: $output_text")

# Example 2: Using stream_to_stdout_cpu
println("\n" * "="^50)
println("Example 2: Streaming Generation")
println("="^50)

# Decode function for streaming
decode_fn = (ids) -> decode_tokens(ids, tokens_data)

println("\nPrompt: \" The\"")
print("Output: ")

# Use streaming generation with sampling
result = stream_to_stdout_cpu(
    model,
    [562],  # " The"
    decode_fn;
    max_tokens=20,
    temperature=0.7f0,
    top_p=0.9f0,
    top_k=40
)

println("\nFull output: $result")
