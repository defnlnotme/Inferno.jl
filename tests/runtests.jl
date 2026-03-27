using Test
using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU
using Inferno.Dequant

const MODEL_PATH = get(ENV, "INFERNO_MODEL_PATH", "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
const MODEL_EXISTS = isfile(MODEL_PATH)

@testset "Inferno.jl CPU Backend" begin
    
    if !MODEL_EXISTS
        @warn "Model not found at $MODEL_PATH - skipping model-dependent tests"
        @warn "Download with: ./scripts/download_model.sh"
    end
    
    @testset "GGUF Loading" begin
        if MODEL_EXISTS
            file = read_gguf(MODEL_PATH)
            
            # Test metadata
            @test haskey(file.metadata, "general.architecture")
            @test file.metadata["general.architecture"] == "qwen35"
            
            # Test tensor presence
            @test haskey(file.tensors, "blk.0.attn_qkv.weight")
            @test haskey(file.tensors, "blk.3.attn_q.weight")
            
            # Test tensor dimensions
            qkv = file.tensors["blk.0.attn_qkv.weight"]
            @test prod(qkv.dimensions) > 0
        else
            @test_skip false
        end
    end
    
    @testset "Model Loading" begin
        if MODEL_EXISTS
            model, file = load_model_cpu(MODEL_PATH)
            
            # Test config
            @test model.config.hidden_size == 1024
            @test model.config.num_hidden_layers == 24
            @test model.config.num_attention_heads == 8
            @test model.config.num_key_value_heads == 2
            @test model.config.head_dim == 256
            
            # Test embedding shape
            @test size(model.embed) == (1024, 248320)
            
            # Test layer count
            @test length(model.layers) == 24
            
            # Test RoPE
            @test model.rope.rotary_dim == 64  # partial rotary
        else
            @test_skip false
        end
    end
    
    @testset "Single Token Inference" begin
        if MODEL_EXISTS
            model, file = load_model_cpu(MODEL_PATH)
            
            # Initialize caches
            caches = [init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
            reset_states_cpu!(model)
            
            # Get embedding for token 0
            x = view(model.embed, :, 1)
            
            # Process through first layer
            x1 = model.layers[1](x, 0, model.rope, caches[1])
            
            @test !any(isnan, x1)
            @test length(x1) == model.config.hidden_size
            
            # Full forward pass
            caches = [init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
            reset_states_cpu!(model)
            
            x = view(model.embed, :, 1)
            for (i, layer) in enumerate(model.layers)
                x = layer(x, 0, model.rope, caches[i])
            end
            x = model.final_norm(x)
            logits = model.lm_head * x
            
            @test !any(isnan, logits)
            @test length(logits) == model.config.vocab_size
            @test maximum(logits) > minimum(logits)
        else
            @test_skip false
        end
    end
    
    @testset "Token Generation" begin
        if MODEL_EXISTS
            model, file = load_model_cpu(MODEL_PATH)
            tokens_data = file.metadata["tokenizer.ggml.tokens"]
            
            # Initialize
            caches = [init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
            reset_states_cpu!(model)
            
            # Generate 5 tokens starting from token 1
            tokens = [1]
            
            for _ in 1:5
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
            
            @test length(tokens) == 6  # 1 initial + 5 generated
            @test all(t -> 1 <= t <= model.config.vocab_size, tokens)
        else
            @test_skip false
        end
    end
    
    @testset "Attention Layer" begin
        if MODEL_EXISTS
            model, file = load_model_cpu(MODEL_PATH)
            
            # Layer 3 is attention (index 4 in 1-based)
            attn_layer = model.layers[4]
            @test !attn_layer.is_ssm
            
            attn = attn_layer.op
            @test attn.n_heads == 8
            @test attn.n_kv == 2
            @test attn.head_dim == 256
            @test size(attn.wq) == (4096, 1024)  # query + gate
            @test size(attn.wk) == (512, 1024)
            @test size(attn.wv) == (512, 1024)
            @test size(attn.wo) == (1024, 2048)
        else
            @test_skip false
        end
    end
    
    @testset "SSM Layer" begin
        if MODEL_EXISTS
            model, file = load_model_cpu(MODEL_PATH)
            
            # Layer 0 is SSM
            ssm_layer = model.layers[1]
            @test ssm_layer.is_ssm
            
            ssm = ssm_layer.op
            @test ssm.num_v_heads == 16
            @test ssm.num_k_heads == 16
            @test ssm.d_inner == 2048
        else
            @test_skip false
        end
    end
end
