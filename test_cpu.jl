using Inferno

function test_cpu()
    println("Testing CPU inference for Qwen3.5...")
    
    # Load model
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Test simple prompt
    prompt = "The capital of France is"
    println("Prompt: ", prompt)
    
    # Tokenize
    tokens = Inferno.Tokenizer.encode(tokenizer, prompt)
    println("Tokens: ", tokens)
    
    # Decode function
    decode_fn = (toks) -> Inferno.Tokenizer.decode(tokenizer, toks)
    
    # Generate with streaming
    println("Generating response...")
    output = ""
    stream = Inferno.generate_stream_cpu(model, tokens, decode_fn; max_tokens=30, temperature=0.0f0, top_p=1.0f0, top_k=1)
    for token in stream
        print(token)
        flush(stdout)
        output *= token
    end
    println()
    
    # Check if "Paris" appears in output
    if occursin("Paris", output) || occursin("paris", lowercase(output))
        println("✓ PASS: Model produced expected output")
        return true
    else
        println("✗ FAIL: Model did not produce expected output")
        println("Output was: ", output)
        return false
    end
end

test_cpu()
