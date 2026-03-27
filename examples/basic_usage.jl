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

# Get tokenizer
tokens_data = file.metadata["tokenizer.ggml.tokens"]

# Initialize KV caches for each layer
caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
ModelCPU.reset_states_cpu!(model)

# Tokenize input (simple word lookup - for production use proper tokenizer)
# "What is" in Qwen tokenizer
input_tokens = [3710, 369]  # "What", " is"

println("\nInput tokens: $input_tokens")
println("Input text: ", join([tokens_data[t+1] for t in input_tokens]))

# Generate tokens
println("\nGenerating...")
all_tokens = copy(input_tokens)

for i in 1:20
    tok = all_tokens[end]
    pos = length(all_tokens) - 1
    
    # Get embedding
    x = view(model.embed, :, tok)
    
    # Forward through all layers
    for (j, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, caches[j])
    end
    
    # Final normalization
    x = model.final_norm(x)
    
    # Compute logits
    logits = model.lm_head * x
    
    # Greedy decode (take argmax)
    next_token = argmax(logits)
    push!(all_tokens, next_token)
    
    # Print progress
    print(".")
end
println()

# Decode output
output_text = join([tokens_data[t+1] for t in all_tokens])
println("\nGenerated text: $output_text")
