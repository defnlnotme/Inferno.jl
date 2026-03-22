using Test
using Inferno
using Statistics
using oneAPI
using LinearAlgebra

const MODEL_PATH = get(ENV, "INFERNO_MODEL", "unsloth/Qwen3.5-0.8B-GGUF")

"""
Comprehensive debugging tests that bisect each step of the inference process.
These tests provide fine-grained control and assurance for each component.
"""

@testset "DEBUGGING: Step-by-Step Inference Analysis" begin

    @testset "STEP 1: Device Selection and GPU Health" begin
        println("\n=== STEP 1: Device Selection and GPU Health ===")

        # Test device enumeration
        devs = collect(oneAPI.devices())
        @test length(devs) > 0
        println("  Found $(length(devs)) GPU devices")

        for (i, dev) in enumerate(devs)
            println("  GPU $i: $(oneAPI.name(dev))")
        end

        # Test device probe function
        probe_result = Inferno.probe_device(devs[1])
        println("  GPU 1 probe result: $probe_result")
        @test probe_result in [:ok, :fast_fail]

        # Test device selection logic
        if length(devs) >= 2
            selected_idx = Inferno.select_device!(devs, 2)
            @test selected_idx == 2
            println("  Successfully selected GPU 2")
        else
            selected_idx = Inferno.select_device!(devs, 1)
            @test selected_idx == 1
            println("  Using only available GPU 1")
        end

        # Test basic GPU memory operations
        try
            test_arr = oneArray(Float16[1.0, 2.0, 3.0])
            test_sum = sum(test_arr)
            @test test_sum ≈ Float16(6.0)
            println("  GPU memory operations: OK")
            oneAPI.unsafe_free!(test_arr)
        catch e
            @test false "GPU memory operations failed: $e"
        end
    end

    @testset "STEP 2: GGUF File Parsing and Validation" begin
        println("\n=== STEP 2: GGUF File Parsing and Validation ===")

        # Test file existence and basic properties
        @test isfile(MODEL_PATH)
        file_size = filesize(MODEL_PATH)
        println("  Model file size: $(file_size ÷ 1024 ÷ 1024) MB")
        @test file_size > 100_000_000  # At least 100MB

        # Test GGUF parsing
        file = Inferno.GGUF.read_gguf(MODEL_PATH)
        @test length(file.metadata) > 0
        @test length(file.tensors) > 0
        @test file.data_offset > 0

        println("  Metadata entries: $(length(file.metadata))")
        println("  Tensor count: $(length(file.tensors))")
        println("  Data offset: $(file.data_offset)")

        # Validate critical metadata
        @test haskey(file.metadata, "general.architecture")
        arch = file.metadata["general.architecture"]
        println("  Architecture: $arch")

        # Validate architecture-specific keys
        arch_keys = [
            "$(arch).vocab_size",
            "$(arch).embedding_length",
            "$(arch).block_count",
            "$(arch).attention.head_count"
        ]

        for key in arch_keys
            @test haskey(file.metadata, key) "Missing critical metadata: $key"
            println("  $key: $(file.metadata[key])")
        end

        # Validate tensor naming convention
        expected_tensors = [
            "token_embd.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_o.weight",
            "output_norm.weight",
            "output.weight"
        ]

        for tensor_name in expected_tensors
            if haskey(file.tensors, tensor_name)
                tensor_info = file.tensors[tensor_name]
                println("  $tensor_name: $(tensor_info.type), dims=$(tensor_info.dimensions)")
                @test prod(tensor_info.dimensions) > 0
            else
                println("  Warning: Missing tensor $tensor_name")
            end
        end
    end

    @testset "STEP 3: Configuration Extraction and Validation" begin
        println("\n=== STEP 3: Configuration Extraction and Validation ===")

        file = Inferno.GGUF.read_gguf(MODEL_PATH)
        arch_str = get(file.metadata, "general.architecture", "qwen2")
        arch = Symbol(arch_str)

        # Test config creation
        config = Inferno.Model.QwenConfig(
            architecture=arch,
            vocab_size=Int(get(file.metadata, "$(arch_str).vocab_size", 32000)),
            hidden_size=Int(get(file.metadata, "$(arch_str).embedding_length", 1024)),
            intermediate_size=Int(get(file.metadata, "$(arch_str).feed_forward_length", 3584)),
            num_hidden_layers=Int(get(file.metadata, "$(arch_str).block_count", 24)),
            num_attention_heads=Int(get(file.metadata, "$(arch_str).attention.head_count", 8)),
            num_key_value_heads=Int(get(file.metadata, "$(arch_str).attention.head_count_kv", 2)),
            head_dim=Int(get(file.metadata, "$(arch_str).attention.key_length", 256)),
            rms_norm_eps=Float16(get(file.metadata, "$(arch_str).attention.layer_norm_rms_epsilon", 1.0e-6)),
            rope_theta=Float16(get(file.metadata, "$(arch_str).rope.freq_base", 10000000.0)),
            max_position_embeddings=min(4096, Int(get(file.metadata, "$(arch_str).context_length", 32768))),
            full_attention_interval=Int(get(file.metadata, "$(arch_str).full_attention_interval", 4)),
            ssm_inner_size=Int(get(file.metadata, "$(arch_str).ssm.inner_size", 2048)),
            ssm_state_size=Int(get(file.metadata, "$(arch_str).ssm.state_size", 128)),
            ssm_group_count=Int(get(file.metadata, "$(arch_str).ssm.group_count", 16)),
            ssm_time_step_rank=Int(get(file.metadata, "$(arch_str).ssm.time_step_rank", 16)),
            num_experts=Int(get(file.metadata, "$(arch_str).expert_count", 0)),
            num_experts_per_tok=Int(get(file.metadata, "$(arch_str).expert_used_count", 0)),
            q_lora_rank=Int(get(file.metadata, "$(arch_str).attention.q_lora_rank", 0)),
            kv_lora_rank=Int(get(file.metadata, "$(arch_str).attention.kv_lora_rank", 0)),
            qk_rope_head_dim=Int(get(file.metadata, "$(arch_str).attention.qk_rope_head_dim", 0)),
            v_head_dim=Int(get(file.metadata, "$(arch_str).attention.v_head_dim", 0)),
        )

        println("  Configuration extracted:")
        println("    Architecture: $(config.architecture)")
        println("    Hidden size: $(config.hidden_size)")
        println("    Layers: $(config.num_hidden_layers)")
        println("    Heads: $(config.num_attention_heads)")
        println("    KV heads: $(config.num_key_value_heads)")
        println("    Vocab size: $(config.vocab_size)")
        println("    Max seq length: $(config.max_position_embeddings)")

        # Validate configuration consistency
        @test config.hidden_size > 0
        @test config.num_hidden_layers > 0
        @test config.num_attention_heads > 0
        @test config.num_key_value_heads > 0
        @test config.vocab_size > 0
        @test config.max_position_embeddings > 0

        # Test head dimension calculation
        expected_head_dim = config.hidden_size ÷ config.num_attention_heads
        if config.head_dim == 0
            config.head_dim = expected_head_dim
        end
        @test config.head_dim == expected_head_dim
        println("    Head dim: $(config.head_dim)")

        # Test KV head consistency
        @test config.num_key_value_heads ≤ config.num_attention_heads
    end

    @testset "STEP 4: Tensor Extraction and Dequantization" begin
        println("\n=== STEP 4: Tensor Extraction and Dequantization ===")

        file = Inferno.GGUF.read_gguf(MODEL_PATH)

        # Test embedding tensor extraction
        println("  Testing embedding tensor...")
        embed_tensor = Inferno.Loader.extract_tensor(file, "token_embd.weight")
        @test size(embed_tensor, 1) > 0
        @test size(embed_tensor, 2) > 0
        println("    Embedding shape: $(size(embed_tensor))")
        println("    Embedding dtype: $(eltype(embed_tensor))")

        # Test layer 0 tensors
        layer_prefix = "blk.0"
        critical_tensors = [
            "$(layer_prefix).attn_qkv.weight",
            "$(layer_prefix).attn_o.weight",
            "$(layer_prefix).attn_norm.weight",
            "$(layer_prefix).ffn_down.weight",
            "$(layer_prefix).ffn_up.weight",
            "$(layer_prefix).ffn_norm.weight"
        ]

        for tensor_name in critical_tensors
            if haskey(file.tensors, tensor_name)
                println("  Testing $tensor_name...")
                try
                    tensor = Inferno.Loader.extract_tensor(file, tensor_name)
                    @test ndims(tensor) ≥ 2
                    @test all(size(tensor) .> 0)
                    println("    Shape: $(size(tensor)), Type: $(eltype(tensor))")

                    # Test for NaN/Inf in extracted tensor
                    if eltype(tensor) <: AbstractFloat
                        finite_count = count(isfinite, tensor)
                        total_count = length(tensor)
                        println("    Finite values: $finite_count/$total_count")
                        @test finite_count == total_count "Found non-finite values in $tensor_name"
                    end
                catch e
                    @test false "Failed to extract $tensor_name: $e"
                end
            else
                println("  Warning: Missing $tensor_name")
            end
        end

        # Test quantization type handling
        println("  Testing quantization types...")
        type_counts = Dict{String, Int}()
        for (name, info) in file.tensors
            type_str = string(info.type)
            type_counts[type_str] = get(type_counts, type_str, 0) + 1
        end

        println("    Quantization type distribution:")
        for (type_str, count) in sort(collect(type_counts))
            println("      $type_str: $count tensors")
        end

        @test any(v -> v > 0, values(type_counts)) "Should have some tensors"
    end

    @testset "STEP 5: Tokenizer Loading and Validation" begin
        println("\n=== STEP 5: Tokenizer Loading and Validation ===")

        file = Inferno.GGUF.read_gguf(MODEL_PATH)
        tok = Inferno.Tokenizer.load_tokenizer(file.metadata)

        # Test basic tokenizer properties
        @test length(tok.id_to_token) > 0
        @test tok.eos_id > 0
        @test tok.bos_id ≥ -1  # BOS might be -1 if missing, or 0+ otherwise

        println("  Tokenizer properties:")
        println("    Vocab size: $(length(tok.id_to_token))")
        println("    BOS ID: $(tok.bos_id)")
        println("    EOS ID: $(tok.eos_id)")

        # Test encoding basic text
        test_texts = [
            "Hello",
            "Hello world",
            "The capital of France is",
            "123",
            "!@#\$%^&*()",
            "Hello\nWorld",
            "  spaces  "
        ]

        for text in test_texts
            println("  Testing encoding: \"$text\"")
            try
                ids = Inferno.Tokenizer.encode(tok, text)
                @test length(ids) > 0
                @test all(id -> 0 ≤ id < length(tok.id_to_token), ids)

                # Test round-trip
                decoded = Inferno.Tokenizer.decode(tok, ids)
                @test length(decoded) > 0
                println("    $(length(ids)) tokens -> \"$decoded\"")

                # Check for obvious encoding errors
                if occursin("", decoded)
                    println("    ⚠ Contains replacement character")
                else
                    println("    ✓ Encoding successful")
                end
            catch e
                @test false "Failed to encode/decode \"$text\": $e"
            end
        end

        # Test special tokens
        if tok.eos_id > 0
            eos_decoded = Inferno.Tokenizer.decode(tok, [tok.eos_id])
            println("  EOS token: $(tok.eos_id) -> \"$eos_decoded\"")
        end

        if tok.bos_id > 0
            bos_decoded = Inferno.Tokenizer.decode(tok, [tok.bos_id])
            println("  BOS token: $(tok.bos_id) -> \"$bos_decoded\"")
        end
    end

    @testset "STEP 6: Model Loading and GPU Memory" begin
        println("\n=== STEP 6: Model Loading and GPU Memory ===")

        # Initialize GPU tables before loading
        Inferno.Model.init_gpu_tables(
            Inferno.QuantsData.IQ2XXS_GRID,
            Inferno.QuantsData.KSIGNS_IQ2XS,
            Inferno.QuantsData.KMASK_IQ2XS
        )
        println("  GPU tables initialized")

        # Load model
        model, tok = Inferno.load_model(MODEL_PATH; device=2)

        # Test model structure
        @test model.config.hidden_size > 0
        @test model.config.num_hidden_layers > 0
        @test length(model.layers) == model.config.num_hidden_layers

        println("  Model loaded successfully:")
        println("    Hidden size: $(model.config.hidden_size)")
        println("    Layers: $(length(model.layers))")
        println("    Heads: $(model.config.num_attention_heads)")

        # Test embedding matrix
        @test size(model.embed.weight, 1) == model.config.hidden_size
        @test size(model.embed.weight, 2) == model.config.vocab_size
        println("    Embedding: $(size(model.embed.weight))")

        # Test layer structures
        for (i, layer) in enumerate(model.layers[1:min(3, end)])  # Test first 3 layers
            println("    Layer $i:")
            @test layer.attn.qkv_proj !== nothing
            @test layer.attn.o_proj !== nothing
            @test layer.attn_norm !== nothing
            @test layer.ffn.down !== nothing
            @test layer.ffn.up !== nothing
            @test layer.ffn_norm !== nothing

            # Test GPU memory allocation
            try
                # Simple test: can we access the weights?
                qkv_size = size(layer.attn.qkv_proj)
                println("      QKV: $(qkv_size)")
                @test all(qkv_size .> 0)
            catch e
                @test false "Layer $i GPU memory access failed: $e"
            end
        end

        # Test normalization layers
        @test model.norm !== nothing
        println("    Output norm: OK")

        # Test final projection
        @test model.proj !== nothing
        proj_size = size(model.proj)
        println("    Final projection: $(proj_size)")
        @test proj_size[1] == model.config.vocab_size
    end

    @testset "STEP 7: KV Cache Initialization" begin
        println("\n=== STEP 7: KV Cache Initialization ===")

        model, tok = Inferno.load_model(MODEL_PATH; device=2)

        # Test KV cache creation
        caches = []
        for i in 1:model.config.num_hidden_layers
            try
                cache = Inferno.Model.init_kv_cache(model.config)
                push!(caches, cache)

                # Test cache properties
                @test cache.k !== nothing
                @test cache.v !== nothing
                @test cache.pos ≥ 0

                if i == 1
                    println("  Cache $i properties:")
                    println("    K shape: $(size(cache.k))")
                    println("    V shape: $(size(cache.v))")
                    println("    Position: $(cache.pos)")
                end
            catch e
                @test false "Failed to initialize KV cache $i: $e"
            end
        end

        @test length(caches) == model.config.num_hidden_layers
        println("  All $(length(caches)) KV caches initialized")
    end

    @testset "STEP 8: Forward Pass - Prefill" begin
        println("\n=== STEP 8: Forward Pass - Prefill ===")

        model, tok = Inferno.load_model(MODEL_PATH; device=2)

        # Initialize caches
        caches = [Inferno.Model.init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

        # Test with simple prompt
        prompt = "The capital of France is"
        tokens = Inferno.Tokenizer.encode(tok, prompt)

        println("  Testing prefill with prompt: \"$prompt\"")
        println("    Tokens: $(length(tokens)) -> $tokens")

        # Forward pass
        try
            logits = Inferno.Model.forward!(model, tokens, 0, caches)

            # Test output shape
            @test size(logits, 1) == model.config.vocab_size
            @test size(logits, 2) == length(tokens)
            println("    Output logits shape: $(size(logits))")

            # Test last token logits (for generation)
            last_logits = vec(logits[:, end])

            # Test for finite values
            finite_count = count(isfinite, last_logits)
            total_count = length(last_logits)
            println("    Finite logits: $finite_count/$total_count")
            @test finite_count == total_count "Found non-finite logits"

            # Test logits statistics
            logits_mean = mean(last_logits)
            logits_max = maximum(last_logits)
            logits_min = minimum(last_logits)
            logits_std = std(last_logits)

            println("    Logits stats:")
            println("      Mean: $(round(logits_mean, digits=3))")
            println("      Max:  $(round(logits_max, digits=3))")
            println("      Min:  $(round(logits_min, digits=3))")
            println("      Std:  $(round(logits_std, digits=3))")

            # Test reasonable value ranges
            @test isfinite(logits_mean)
            @test logits_max > logits_min
            @test logits_max < Float16(1000.0) "Logits too large"
            @test logits_min > -Float16(1000.0) "Logits too small"
            @test logits_std > 0 "Logits have no variance"

        catch e
            @test false "Forward pass failed: $e"
        end
    end

    @testset "STEP 9: Forward Pass - Generation Step" begin
        println("\n=== STEP 9: Forward Pass - Generation Step ===")

        model, tok = Inferno.load_model(MODEL_PATH; device=2)

        # Initialize caches and do prefill
        caches = [Inferno.Model.init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

        prompt = "The capital of France is"
        tokens = Inferno.Tokenizer.encode(tok, prompt)

        # Prefill
        logits = Inferno.Model.forward!(model, tokens, 0, caches)
        last_logits = vec(logits[:, end])

        # Sample first token
        first_token_id = argmax(last_logits)
        first_token_str = Inferno.Tokenizer.decode(tok, [first_token_id])

        println("  First generation step:")
        println("    Sampled token: $first_token_id -> \"$first_token_str\"")

        # Generation step
        try
            current_pos = caches[1].pos
            gen_logits = Inferno.Model.forward!(model, [first_token_id], current_pos, caches)

            # Test generation output
            @test size(gen_logits, 1) == model.config.vocab_size
            @test size(gen_logits, 2) == 1  # Single token generation
            println("    Generation logits shape: $(size(gen_logits))")

            gen_last_logits = vec(gen_logits[:, 1])

            # Test finite values
            finite_count = count(isfinite, gen_last_logits)
            total_count = length(gen_last_logits)
            println("    Finite logits: $finite_count/$total_count")
            @test finite_count == total_count

            # Test statistics
            gen_mean = mean(gen_last_logits)
            gen_max = maximum(gen_last_logits)
            gen_min = minimum(gen_last_logits)

            println("    Generation logits stats:")
            println("      Mean: $(round(gen_mean, digits=3))")
            println("      Max:  $(round(gen_max, digits=3))")
            println("      Min:  $(round(gen_min, digits=3))")

            @test isfinite(gen_mean)
            @test gen_max > gen_min

        catch e
            @test false "Generation step failed: $e"
        end
    end

    @testset "STEP 10: Sampling and Token Generation" begin
        println("\n=== STEP 10: Sampling and Token Generation ===")

        # Test sampling function directly
        test_logits = [
            Float16[1.0, 5.0, 2.0, 4.0, 3.0],  # Clear max at index 2
            Float16[-1.0, -5.0, -2.0, -4.0],  # Clear max at index 1 (least negative)
            Float16[0.1, 0.1, 0.1, 0.1],      # All equal
        ]

        for (i, logits) in enumerate(test_logits)
            println("  Test sampling case $i:")
            println("    Input logits: $logits")

            # Test greedy sampling (temp=0)
            greedy_result = Inferno.Engine.sample(logits, Float16(0.0), Float16(1.0))
            expected_argmax = argmax(logits)
            println("    Greedy result: $greedy_result (expected $expected_argmax)")
            @test greedy_result == expected_argmax

            # Test temperature sampling
            temp_result = Inferno.Engine.sample(logits, Float16(0.7), Float16(0.9))
            println("    Temperature result: $temp_result")
            @test 1 ≤ temp_result ≤ length(logits)
        end

        # Test with model logits
        model, tok = Inferno.load_model(MODEL_PATH; device=2)
        caches = [Inferno.Model.init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

        prompt = "Hello"
        tokens = Inferno.Tokenizer.encode(tok, prompt)
        logits = Inferno.Model.forward!(model, tokens, 0, caches)
        last_logits = vec(logits[:, end])

        println("  Model logits sampling:")
        println("    Vocab size: $(length(last_logits))")

        # Test different sampling strategies
        strategies = [
            ("greedy", Float16(0.0), Float16(1.0)),
            ("low_temp", Float16(0.3), Float16(0.9)),
            ("med_temp", Float16(0.7), Float16(0.8)),
            ("high_temp", Float16(1.0), Float16(0.9)),
        ]

        for (name, temp, top_p) in strategies
            try
                token_id = Inferno.Engine.sample(last_logits, temp, top_p)
                token_str = Inferno.Tokenizer.decode(tok, [token_id])
                println("    $name: $token_id -> \"$token_str\"")
                @test 1 ≤ token_id ≤ length(tok.id_to_token)
                @test length(token_str) > 0
            catch e
                @test false "Sampling strategy $name failed: $e"
            end
        end
    end

    @testset "STEP 11: End-to-End Generation Validation" begin
        println("\n=== STEP 11: End-to-End Generation Validation ===")

        model, tok = Inferno.load_model(MODEL_PATH; device=2)

        test_prompts = [
            "The capital of France is",
            "1+1=",
            "Hello",
            "A",
        ]

        for prompt in test_prompts
            println("  Testing E2E generation: \"$prompt\"")

            try
                # Generate 5 tokens
                generated_tokens = Int[]
                generated_text = ""

                # Initialize
                tokens = Inferno.Tokenizer.encode(tok, prompt)
                caches = [Inferno.Model.init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

                # Prefill
                logits = Inferno.Model.forward!(model, tokens, 0, caches)
                last_logits = vec(logits[:, end])

                # Generate tokens
                for step in 1:5
                    token_id = Inferno.Engine.sample(last_logits, Float16(0.7), Float16(0.8))
                    token_str = Inferno.Tokenizer.decode(tok, [token_id])

                    push!(generated_tokens, token_id)
                    generated_text *= token_str

                    # Stop at EOS
                    if token_id == tok.eos_id
                        println("    EOS reached at step $step")
                        break
                    end

                    # Next step
                    current_pos = caches[1].pos
                    next_logits = Inferno.Model.forward!(model, [token_id], current_pos, caches)
                    last_logits = vec(next_logits[:, 1])
                end

                println("    Generated $(length(generated_tokens)) tokens: \"$generated_text\"")
                @test length(generated_tokens) > 0
                @test length(generated_text) > 0

                # Check for obvious errors
                if occursin("", generated_text)
                    println("    ⚠ Contains replacement character")
                else
                    println("    ✓ Generation successful")
                end

            catch e
                @test false "E2E generation failed for \"$prompt\": $e"
            end
        end
    end

    @testset "STEP 12: Memory and Performance Monitoring" begin
        println("\n=== STEP 12: Memory and Performance Monitoring ===")

        model, tok = Inferno.load_model(MODEL_PATH; device=2)

        # Test memory usage estimation
        println("  Memory usage:")

        # Estimate model parameters
        total_params = 0
        for layer in model.layers
            # Rough estimation (this is approximate due to quantization)
            qkv_size = prod(size(layer.attn.qkv_proj))
            o_size = prod(size(layer.attn.o_proj))
            ffn_up_size = prod(size(layer.ffn.up))
            ffn_down_size = prod(size(layer.ffn.down))
            total_params += qkv_size + o_size + ffn_up_size + ffn_down_size
        end

        embed_size = prod(size(model.embed.weight))
        proj_size = prod(size(model.proj))
        total_params += embed_size + proj_size

        println("    Estimated parameters: $(total_params ÷ 1_000_000)M")

        # Test forward pass timing
        prompt = "The capital of France is"
        tokens = Inferno.Tokenizer.encode(tok, prompt)
        caches = [Inferno.Model.init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

        # Time prefill
        prefill_time = @elapsed begin
            logits = Inferno.Model.forward!(model, tokens, 0, caches)
        end

        println("    Prefill time: $(round(prefill_time * 1000, digits=2))ms")
        @test prefill_time < 10.0 "Prefill too slow: $(prefill_time)s"

        # Time generation step
        gen_time = @elapsed begin
            token_id = argmax(vec(logits[:, end]))
            current_pos = caches[1].pos
            gen_logits = Inferno.Model.forward!(model, [token_id], current_pos, caches)
        end

        println("    Generation step time: $(round(gen_time * 1000, digits=2))ms")
        @test gen_time < 1.0 "Generation too slow: $(gen_time)s"

        # Test tokens/second
        tokens_per_sec = length(tokens) / prefill_time
        println("    Prefill tokens/sec: $(round(tokens_per_sec, digits=1))")

        gen_tokens_per_sec = 1.0 / gen_time
        println("    Generation tokens/sec: $(round(gen_tokens_per_sec, digits=1))")
    end
end

println("\n=== DEBUGGING TESTS COMPLETE ===")
println("All steps have been validated. Check output above for any warnings or failures.")
