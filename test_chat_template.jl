using Inferno
using LinearAlgebra

function test_with_chat_template()
    model, tokenizer = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    
    # Try with chat template: <|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n
    # But since we just want to continue, let's try: The capital of France is
    # Or with system prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n
    
    # First, try without any special tokens (raw prompt)
    prompt_tokens = [761, 6512, 315, 9339, 370]  # "The capital of France is"
    
    # Now try with the <|im_start|> token prepended
    # <|im_start|> = 248046
    prompt_with_im = vcat([248046], prompt_tokens)
    
    println("=== Without special tokens ===")
    x = process_prompt(model, prompt_tokens, caches)
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    println("Top prediction: ", argmax(logits) - 1)
    println("Logit for '[': ", round(logits[60], digits=3))
    
    # Reset
    Inferno.ModelCPU.reset_states_cpu!(model)
    for c in caches
        fill!(c.k, 0.0f0)
        fill!(c.v, 0.0f0)
    end
    
    println("\n=== With <|im_start|> token ===")
    x = process_prompt(model, prompt_with_im, caches)
    x_normed = model.final_norm(x)
    logits = model.lm_head * x_normed
    println("Top prediction: ", argmax(logits) - 1)
    println("Logit for '[': ", round(logits[60], digits=3))
end

function process_prompt(model, tokens, caches)
    x = zeros(Float32, model.config.hidden_size)
    for (pos, tok) in enumerate(tokens)
        x = model.embed[:, tok]
        for (i, layer) in enumerate(model.layers)
            x = layer(x, pos-1, model.rope, caches[i])
        end
    end
    return x
end

test_with_chat_template()
