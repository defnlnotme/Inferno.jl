using Inferno
using LinearAlgebra
using Base.Threads

function profile_ssm()
    model, tok = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B/")
    
    prompt_tokens = Inferno.Tokenizer.encode(tok, "test")
    
    # Warm up
    Inferno.ModelCPU.reset_states_cpu!(model)
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    _ = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
    
    println("=== SSM Layer Detailed Profile ===")
    println()
    
    # Find an SSM layer
    ssm_layer_idx = findfirst(l -> l.is_ssm, model.layers)
    ssm_layer = model.layers[ssm_layer_idx]
    ssm = ssm_layer.op
    
    println("SSM layer $ssm_layer_idx:")
    println("  in_proj size: ", size(ssm.in_proj))
    println("  conv1d kernel size: ", size(ssm.conv1d_kernel))
    println("  A_log size: ", size(ssm.A_log))
    println("  out_proj size: ", size(ssm.out_proj))
    println()
    
    # Time individual operations
    h = model.embed[:, prompt_tokens[end]]
    pos = length(prompt_tokens)
    
    # Norm
    norm_time = @elapsed Inferno.ModelCPU.rmsnorm_cpu!(ssm_layer.norm_buf1, h, ssm_layer.in_norm)
    println("Input norm: $(round(norm_time * 1000, digits=3)) ms")
    
    h_norm = ssm_layer.norm_buf1
    
    # in_proj
    in_proj_time = @elapsed begin
        mul!(ssm.x_proj, ssm.in_proj, h_norm)
    end
    println("in_proj: $(round(in_proj_time * 1000, digits=3)) ms")
    
    # Split projections
    split_time = @elapsed begin
        x = view(ssm.x_proj, 1:ssm.in_proj_split[1], :)
        z = view(ssm.x_proj, ssm.in_proj_split[1]+1:ssm.in_proj_split[2], :_)
        # ... rest of SSM
    end
    println("Split views: $(round(split_time * 1000, digits=3)) ms")
    
    # Conv1d
    conv_time = @elapsed begin
        # Conv1d update
        # ... 
    end
    
    println()
    println("=== Timing full SSM layer ===")
    
    times = Float64[]
    for i in 1:20
        h = model.embed[:, prompt_tokens[end]]
        t0 = time()
        h = ssm_layer(h, pos, model.rope, caches[ssm_layer_idx])
        t1 = time()
        push!(times, t1 - t0)
    end
    
    println("Full SSM layer:")
    println("  Mean: $(round(sum(times)/length(times)*1000, digits=2)) ms")
    println("  Min:  $(round(minimum(times)*1000, digits=2)) ms")
end

profile_ssm()
