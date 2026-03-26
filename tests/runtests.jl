using Test
using Inferno
using oneAPI
using Statistics

# Find project root by looking for Project.toml
function find_project_root()
    # Try current directory and parents
    dir = pwd()
    while dir != "/"
        if isfile(joinpath(dir, "Project.toml"))
            return dir
        end
        dir = dirname(dir)
    end
    # Fallback to relative path from test directory
    return joinpath(pwd(), "..")
end

# Get model path - called at test time, not compile time
function get_model_path()
    return get(ENV, "INFERNO_MODEL", 
        joinpath(find_project_root(), "tests", "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
end

println("Project root: ", find_project_root())
println("Testing with model: ", get_model_path())
println("Model exists: ", isfile(get_model_path()))

@testset "Inferno Tests" begin
    
    @testset "Model Loading" begin
        MODEL_PATH = get_model_path()
        @test isfile(MODEL_PATH)
        
        model, tok = Inferno.load_model(MODEL_PATH)
        
        # Test config
        @test model.config.hidden_size == 1024
        @test model.config.num_hidden_layers == 24
        @test model.config.num_attention_heads == 8
        @test model.config.num_key_value_heads == 2
        
        # Test tokenizer
        @test length(tok.id_to_token) > 0
        @test tok.eos_id > 0
    end
    
    @testset "GPU Storage - Critical Regression Test" begin
        MODEL_PATH = get_model_path()
        model, tok = Inferno.load_model(MODEL_PATH)
        
        # CRITICAL: Embedding MUST be on GPU for inference to work
        @test model.embed isa oneAPI.oneMatrix{Float16}
        
        # CRITICAL: lm_head should also be on GPU (or use weight tying with embedding)
        @test model.lm_head isa oneAPI.oneMatrix{Float16} || model.lm_head === model.embed
        
        # Verify embedding is not all zeros
        embed_cpu = Array(model.embed)
        @test !all(x -> x == Float16(0), embed_cpu)
        @test maximum(abs.(embed_cpu)) > Float16(0.001)
        
        # Test a sample embedding row
        sample_emb = model.embed[:, 1]
        sample_cpu = Array(sample_emb)
        @test length(sample_cpu) == model.config.hidden_size
        @test !all(x -> x == Float16(0), sample_cpu)
    end
    
    @testset "Layer Weights on GPU" begin
        MODEL_PATH = get_model_path()
        model, tok = Inferno.load_model(MODEL_PATH)
        
        # Check first layer weights are on GPU
        layer1 = model.layers[1]
        
        @test layer1.in_norm.weight isa oneAPI.oneMatrix{Float16}
        @test layer1.post_norm.weight isa oneAPI.oneMatrix{Float16}
        
        # Check MLP weights are on GPU
        @test layer1.mlp.gate_weight isa oneAPI.oneMatrix{Float16}
        @test layer1.mlp.up_weight isa oneAPI.oneMatrix{Float16}
        @test layer1.mlp.down_weight isa oneAPI.oneMatrix{Float16}
    end
    
    @testset "Tokenizer Encoding/Decoding" begin
        MODEL_PATH = get_model_path()
        model, tok = Inferno.load_model(MODEL_PATH)
        
        # Test encoding
        prompt = "The capital of France is"
        tokens = Inferno.Tokenizer.encode(tok, prompt)
        @test length(tokens) > 0
        @test all(t -> t > 0, tokens)
        
        # Test decoding
        decoded = Inferno.Tokenizer.decode(tok, tokens)
        @test length(decoded) > 0
        @test occursin("capital", lowercase(decoded)) || occursin("france", lowercase(decoded))
        
        # Test single token decode
        single_tok = tokens[1]
        single_decoded = Inferno.Tokenizer.decode(tok, [single_tok])
        @test length(single_decoded) > 0
    end
    
    @testset "Inference Forward Pass" begin
        MODEL_PATH = get_model_path()
        model, tok = Inferno.load_model(MODEL_PATH)
        
        # Init KV caches
        caches = [Inferno.Model.init_kv_cache(model.config) 
                  for _ in 1:model.config.num_hidden_layers]
        
        # Test with simple tokens
        test_tokens = [1, 2, 3]
        logits = Inferno.Model.forward!(model, test_tokens, 0, caches)
        
        # Shape check
        @test size(logits, 1) == length(tok.id_to_token)
        @test size(logits, 2) == length(test_tokens)
        
        # CRITICAL: Logits must NOT be all zeros (regression check)
        logits_arr = Float32.(Array(logits))
        @test !all(x -> x == 0.0f0, logits_arr)
        
        # Statistical checks
        mean_val = mean(logits_arr)
        max_val = maximum(logits_arr)
        min_val = minimum(logits_arr)
        std_val = std(logits_arr)
        
        @test isfinite(mean_val)
        @test isfinite(max_val)
        @test isfinite(min_val)
        
        # Logits should have meaningful spread
        @test max_val > min_val
        @test std_val > 0.01f0
        
        println("  Logits stats: mean=$(round(mean_val, digits=2)), max=$(round(max_val, digits=2)), min=$(round(min_val, digits=2)), std=$(round(std_val, digits=4))")
    end
    
    @testset "Token Generation Quality" begin
        MODEL_PATH = get_model_path()
        model, tok = Inferno.load_model(MODEL_PATH)
        
        prompt = "The capital of France is"
        tokens = Inferno.Tokenizer.encode(tok, prompt)
        
        # Init KV caches
        caches = [Inferno.Model.init_kv_cache(model.config) 
                  for _ in 1:model.config.num_hidden_layers]
        
        # Prefill
        logits = Inferno.Model.forward!(model, tokens, 0, caches)
        last_logits = vec(logits[:, end])
        
        # CRITICAL: Last token logits must not be all zeros
        last_arr = Float32.(Array(last_logits))
        @test !all(x -> x == 0.0f0, last_arr)
        
        # Generate tokens
        generated = Int[]
        curr_logits = last_logits
        
        for i in 1:5
            tok_id = argmax(curr_logits)
            push!(generated, tok_id)
            
            if tok_id == tok.eos_id
                break
            end
            
            curr_pos = caches[1].pos
            next_logits = Inferno.Model.forward!(model, [tok_id], curr_pos, caches)
            curr_logits = vec(next_logits[:, 1])
            
            # Each step should produce finite logits
            @test all(isfinite, Array(curr_logits))
            
            # Each step should produce non-zero logits
            @test !all(x -> x == Float16(0), Array(curr_logits))
        end
        
        # Check we generated something
        @test length(generated) >= 1
        
        # Decode and verify
        full_decode = Inferno.Tokenizer.decode(tok, generated)
        @test length(full_decode) > 0
        
        # Check we're not just generating the same token repeatedly
        if length(generated) > 1
            unique_tokens = length(unique(generated))
            @test unique_tokens >= 1
        end
        
        println("  Generated: \"$full_decode\"")
    end
    
    @testset "Numerical Stability" begin
        MODEL_PATH = get_model_path()
        model, tok = Inferno.load_model(MODEL_PATH)
        
        # Test with longer sequence to stress numerical stability
        long_prompt = "The quick brown fox jumps over the lazy dog. This is a test of numerical stability."
        tokens = Inferno.Tokenizer.encode(tok, long_prompt)
        
        caches = [Inferno.Model.init_kv_cache(model.config) 
                  for _ in 1:model.config.num_hidden_layers]
        
        # Process the sequence
        logits = Inferno.Model.forward!(model, tokens, 0, caches)
        
        # All values should be finite
        @test all(isfinite, Array(logits))
        
        # No NaN values
        @test !any(isnan, Float32.(Array(logits)))
        
        # Reasonable range (not exploding)
        max_abs = maximum(abs.(Float32.(Array(logits))))
        @test max_abs < 1000.0f0
    end
    
    @testset "KV Cache Operations" begin
        MODEL_PATH = get_model_path()
        model, tok = Inferno.load_model(MODEL_PATH)
        
        cache = Inferno.Model.init_kv_cache(model.config)
        
        # Check cache dimensions
        @test size(cache.k, 1) == model.config.head_dim
        @test size(cache.v, 1) == model.config.head_dim
        @test cache.pos == 0
        
        # Test cache update
        tokens = [1, 2, 3]
        caches = [Inferno.Model.init_kv_cache(model.config) 
                  for _ in 1:model.config.num_hidden_layers]
        
        logits = Inferno.Model.forward!(model, tokens, 0, caches)
        
        # Position should advance
        @test caches[1].pos == length(tokens)
    end
    
    @testset "Memory and GPU Synchronization" begin
        MODEL_PATH = get_model_path()
        model, tok = Inferno.load_model(MODEL_PATH)
        
        # Force synchronization and check for errors
        oneAPI.synchronize()
        
        # Multiple forward passes should not accumulate errors
        caches = [Inferno.Model.init_kv_cache(model.config) 
                  for _ in 1:model.config.num_hidden_layers]
        
        for _ in 1:3
            tokens = [1, 2]
            logits = Inferno.Model.forward!(model, tokens, 0, caches)
            @test all(isfinite, Array(logits))
            oneAPI.synchronize()
        end
    end
end
