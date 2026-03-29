using Inferno

function test_gpu()
    println("Testing GPU inference for Qwen3.5...")
    
    # Load model - use quantized model
    model, tokenizer = Inferno.load_model("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; device=2)
    
    # Test simple prompt
    prompt = "The capital of France is"
    println("Prompt: ", prompt)
    
    # Generate with streaming
    println("Generating response...")
    output = ""
    stream = Inferno.generate_stream(model, tokenizer, prompt; max_tokens=30, temperature=Float16(0.0), top_p=Float16(1.0), top_k=1)
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

test_gpu()
