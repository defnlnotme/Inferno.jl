# Basic usage example for Inferno.jl CPU backend
# Run with: julia --project=. examples/basic_usage.jl

using Inferno

# Path to your GGUF model
const MODEL_PATH = get(ENV, "INFERNO_MODEL_PATH", "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("Loading model from: $MODEL_PATH")
model, tokenizer = LoaderCPU.load_model_cpu(MODEL_PATH)
println("Model loaded successfully!")
println("  Hidden size: $(model.config.hidden_size)")
println("  Layers: $(model.config.num_hidden_layers)")
println("  Vocab size: $(model.config.vocab_size)")

# Example 1: Manual token-by-token generation
println("\n" * "="^50)
println("Example 1: Manual Generation")
println("="^50)

# Use the BPETokenizer to encode/decode
prompt_tokens = Tokenizer.encode(tokenizer, "The")
println("Starting with prompt tokens: $(prompt_tokens)")

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

output_text = Tokenizer.decode(tokenizer, tokens)
println("\nGenerated: $output_text")

# Example 2: Using stream_to_stdout_cpu
println("\n" * "="^50)
println("Example 2: Streaming Generation")
println("="^50)

# Use the convenience method that handles tokenization internally
println("\nPrompt: \"The\"")
print("Output: ")

result = stream_to_stdout_cpu(
    model,
    tokenizer,
    "The";
    max_tokens=20,
    temperature=0.7f0,
    top_p=0.9f0,
    top_k=40
)

println("\nFull output: $result")
