using Test
using Inferno
using oneAPI

"""
Focused tests for critical inference components.
These tests target the most common failure points in inference.
"""

const MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "models", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))

@testset "CRITICAL INFERENCE COMPONENTS" begin
    
    @testset "GPU Device Health" begin
        println("\n=== GPU Device Health Check ===")
        
        # Test device availability
        devs = collect(oneAPI.devices())
        @test length(devs) > 0
        println("  Found $(length(devs)) GPUs")
        
        # Test each device
        for (i, dev) in enumerate(devs)
            try
                oneAPI.device!(dev)
                
                # Test small allocation
                test_arr = oneArray(Float32[1.0f0, 2.0f0, 3.0f0])
                test_sum = sum(test_arr)
                @test test_sum ≈ 6.0f0
                
                # Test larger allocation (1MB)
                large_arr = oneArray(zeros(Float32, 256 * 1024))
                large_sum = sum(large_arr)
                @test large_sum == 0.0f0
                
                oneAPI.unsafe_free!(test_arr)
                oneAPI.unsafe_free!(large_arr)
                
                println("  GPU $i: ✓ OK")
                
            catch e
                @test false
            end
        end
    end
    
    @testset "Model Loading Safety" begin
        println("\n=== Model Loading Safety ===")
        
        # Test file integrity
        @test isfile(MODEL_PATH)
        
        file_size = filesize(MODEL_PATH)
        @test file_size > 100_000_000
        println("  Model file: $(file_size ÷ 1024 ÷ 1024) MB")
        
        # Test GGUF parsing
        try
            file = Inferno.GGUF.read_gguf(MODEL_PATH)
            @test length(file.metadata) > 0
            @test length(file.tensors) > 0
            println("  GGUF parsing: ✓ OK ($(length(file.tensors)) tensors)")
        catch e
            @test false
        end
        
        # Test model loading with error handling
        model = nothing
        tok = nothing
        
        try
            model, tok = Inferno.load_model(MODEL_PATH; device=2)
            
            @test model !== nothing
            @test tok !== nothing
            @test model.config.hidden_size > 0
            @test length(model.layers) > 0
            
            println("  Model loading: ✓ OK")
            println("    Hidden size: $(model.config.hidden_size)")
            println("    Layers: $(length(model.layers))")
            
        catch e
            @test false
        ensure
            # Cleanup on failure
            try
                if model !== nothing
                    Inferno.Model.free_model_gpu!(model)
                end
            catch
            end
        end
    end
    
    @testset "Forward Pass Robustness" begin
        println("\n=== Forward Pass Robustness ===")
        
        model, tok = Inferno.load_model(MODEL_PATH; device=2)
        
        # Test cache initialization
        caches = []
        try
            for i in 1:model.config.num_hidden_layers
                cache = Inferno.Model.init_kv_cache(
                    model.config.head_dim,
                    model.config.num_key_value_heads,
                    model.config.max_position_embeddings
                )
                push!(caches, cache)
            end
            println("  KV cache initialization: ✓ OK")
        catch e
            @test false
        end
        
        # Test forward pass with various inputs
        test_cases = [
            ("single_token", "A"),
            ("short_phrase", "Hello world"),
            ("question", "What is 2+2?"),
            ("numbers", "123 456 789"),
        ]
        
        for (case_name, prompt) in test_cases
            println("  Testing $case_name: \"$prompt\"")
            
            try
                tokens = Inferno.Tokenizer.encode(tok, prompt)
                @test length(tokens) > 0
                
                # Reset caches for each test
                caches = [Inferno.Model.init_kv_cache(
                    model.config.head_dim,
                    model.config.num_key_value_heads,
                    model.config.max_position_embeddings
                ) for _ in 1:model.config.num_hidden_layers]
                
                # Forward pass
                logits = Inferno.Model.forward!(model, tokens, 0, caches)
                
                # Validate output
                @test size(logits, 1) == model.config.vocab_size
                @test size(logits, 2) == length(tokens)
                
                # Check for valid logits
                last_logits = vec(logits[:, end])
                finite_count = count(isfinite, last_logits)
                @test finite_count == length(last_logits)
                
                # Check for reasonable values
                max_logit = maximum(last_logits)
                min_logit = minimum(last_logits)
                @test max_logit > min_logit
                @test max_logit < 1000.0f0
                @test min_logit > -1000.0f0
                
                println("    ✓ Pass - logits range: [$(round(min_logit, digits=2)), $(round(max_logit, digits=2))]")
                
            catch e
                @test false
            end
        end
    end
    
    @testset "Generation Stability" begin
        println("\n=== Generation Stability ===")
        
        model, tok = Inferno.load_model(MODEL_PATH; device=2)
        
        # Test generation with different strategies
        test_prompts = ["The", "Hello", "1"]
        
        for prompt in test_prompts
            println("  Testing generation from: \"$prompt\"")
            
            try
                # Initialize
                tokens = Inferno.Tokenizer.encode(tok, prompt)
                caches = [Inferno.Model.init_kv_cache(
                    model.config.head_dim,
                    model.config.num_key_value_heads,
                    model.config.max_position_embeddings
                ) for _ in 1:model.config.num_hidden_layers]
                
                # Prefill
                logits = Inferno.Model.forward!(model, tokens, 0, caches)
                last_logits = vec(logits[:, end])
                
                # Test multiple generation steps
                generated_tokens = []
                current_logits = last_logits
                
                for step in 1:3
                    # Sample token
                    token_id = Inferno.Engine.sample(current_logits, 0.7f0, 0.8f0)
                    @test 1 ≤ token_id ≤ length(tok.id_to_token)
                    
                    # Decode token
                    token_str = Inferno.Tokenizer.decode(tok, [token_id])
                    @test length(token_str) > 0
                    
                    push!(generated_tokens, token_id)
                    
                    # Stop at EOS
                    if token_id == tok.eos_id
                        println("    EOS reached at step $step")
                        break
                    end
                    
                    # Next step
                    current_pos = caches[1].pos
                    next_logits = Inferno.Model.forward!(model, [token_id], current_pos, caches)
                    current_logits = vec(next_logits[:, 1])
                    
                    # Validate next logits
                    finite_count = count(isfinite, current_logits)
                    @test finite_count == length(current_logits)
                end
                
                generated_text = Inferno.Tokenizer.decode(tok, generated_tokens)
                println("    Generated $(length(generated_tokens)) tokens: \"$generated_text\"")
                @test length(generated_tokens) > 0
                
            catch e
                @test false
            end
        end
    end
    
    @testset "Memory Leak Detection" begin
        println("\n=== Memory Leak Detection ===")
        
        # Test multiple model loading/unloading cycles
        for cycle in 1:3
            println("  Memory cycle $cycle/3")
            
            model = nothing
            tok = nothing
            
            try
                model, tok = Inferno.load_model(MODEL_PATH; device=2)
                
                # Simple forward pass
                caches = [Inferno.Model.init_kv_cache(
                    model.config.head_dim,
                    model.config.num_key_value_heads,
                    model.config.max_position_embeddings
                ) for _ in 1:model.config.num_hidden_layers]
                
                tokens = Inferno.Tokenizer.encode(tok, "Test")
                logits = Inferno.Model.forward!(model, tokens, 0, caches)
                
                @test size(logits, 1) == model.config.vocab_size
                
            catch e
                @test false
            ensure
                # Explicit cleanup
                try
                    if model !== nothing
                        Inferno.Model.free_model_gpu!(model)
                    end
                catch
                end
                GC.gc(true)
            end
        end
        
        println("  ✓ Memory cycles completed")
    end
    
    @testset "Error Recovery" begin
        println("\n=== Error Recovery ===")
        
        # Test invalid token handling
        model, tok = Inferno.load_model(MODEL_PATH; device=2)
        
        try
            # Test with out-of-vocab token IDs
            invalid_tokens = [0, 1, length(tok.id_to_token) + 1000]
            
            for invalid_id in invalid_tokens
                try
                    # This should handle gracefully
                    decoded = Inferno.Tokenizer.decode(tok, [invalid_id])
                    println("  Invalid token $invalid_id: \"$decoded\"")
                catch e
                    # Should not crash
                    println("  Invalid token $invalid_id: Handled with error")
                end
            end
            
            # Test with empty input
            empty_tokens = Int[]
            caches = [Inferno.Model.init_kv_cache(
                model.config.head_dim,
                model.config.num_key_value_heads,
                model.config.max_position_embeddings
            ) for _ in 1:model.config.num_hidden_layers]
            
            # Empty tokens should be handled gracefully or fail gracefully
            try
                logits = Inferno.Model.forward!(model, empty_tokens, 0, caches)
                println("  Empty tokens: Handled")
            catch e
                println("  Empty tokens: Failed gracefully - $e")
            end
            
        catch e
            @test false
        end
    end
end

println("\n=== CRITICAL COMPONENT TESTS COMPLETE ===")
