#!/usr/bin/env julia
# Detailed step-by-step debugging

using LinearAlgebra
using Statistics
using Inferno
using Inferno.GGUF
using Inferno.ModelCPU
using Inferno.LoaderCPU
using Inferno.Dequant

function main()
    println("=== Loading model ===")
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

    println("\n=== Testing single token forward pass ===")

    # Reset states
    reset_states_cpu!(model)

    # Get token ID for "The"
    tokens = file.metadata["tokenizer.ggml.tokens"]
    
    # Use "The" token
    the_token_id = 561  # ĠThe
    println("Using token id=$the_token_id: '$(tokens[the_token_id+1])'")

    # Initialize caches
    caches = [init_kv_cache_cpu(model.config, 1) for _ in model.layers]

    # Step 1: Get embedding
    println("\n--- Embedding ---")
    x = collect(view(model.embed, :, the_token_id))
    println("Embedding norm: $(norm(x))")
    println("Embedding stats: min=$(minimum(x)), max=$(maximum(x)), mean=$(mean(x))")

    # Step 2: Process through each layer
    for i in 1:length(model.layers)
        layer = model.layers[i]
        println("\n--- Layer $i ($(layer.is_ssm ? "SSM" : "Attention")) ---")
        
        # Input norm
        x_norm = layer.in_norm(x)
        println("After input norm: norm=$(norm(x_norm)), min=$(minimum(x_norm)), max=$(maximum(x_norm))")
        
        # Check for issues
        if isnan(norm(x_norm)) || isinf(norm(x_norm))
            println("ERROR: Invalid values after input norm!")
            break
        end
        
        # Forward through attention/SSM
        residual = layer.op(x_norm, 0, model.rope, caches[i])
        println("After $(layer.is_ssm ? "SSM" : "Attention"): norm=$(norm(residual)), min=$(minimum(residual)), max=$(maximum(residual))")
        
        if isnan(norm(residual)) || isinf(norm(residual))
            println("ERROR: Invalid values after $(layer.is_ssm ? "SSM" : "Attention")!")
            break
        end
        
        # Residual
        x = x + residual
        println("After residual: norm=$(norm(x))")
        
        # Post norm
        x_norm = layer.post_norm(x)
        println("After post norm: norm=$(norm(x_norm))")
        
        # MLP
        residual = layer.mlp(x_norm)
        println("After MLP: norm=$(norm(residual)), min=$(minimum(residual)), max=$(maximum(residual))")
        
        if isnan(norm(residual)) || isinf(norm(residual))
            println("ERROR: Invalid values after MLP!")
            break
        end
        
        # Final residual
        x = x + residual
        println("Layer $i output norm: $(norm(x))")
        
        # Check for growth
        if norm(x) > 100
            println("WARNING: Large hidden state norm!")
        end
    end

    # Final norm
    println("\n--- Final Norm ---")
    x_final = model.final_norm(x)
    println("Final norm output: norm=$(norm(x_final)), min=$(minimum(x_final)), max=$(maximum(x_final))")

    # LM head
    println("\n--- LM Head ---")
    logits = model.lm_head * x_final
    println("Logits stats: min=$(minimum(logits)), max=$(maximum(logits)), mean=$(mean(logits))")
    println("Logits norm: $(norm(logits))")

    # Top tokens
    sorted_idx = sortperm(logits, rev=true)
    println("\nTop 10 tokens:")
    for i in 1:10
        idx = sorted_idx[i]
        println("  $i: id=$idx logit=$(logits[idx]) token='$(tokens[idx+1])'")
    end

    # Softmax
    max_logit = maximum(logits)
    exp_logits = exp.(logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)
    println("\nTop 10 probs:")
    sorted_probs = sortperm(probs, rev=true)
    for i in 1:10
        idx = sorted_probs[i]
        println("  $i: id=$idx prob=$(probs[idx]) token='$(tokens[idx+1])'")
    end
end

main()
