using Inferno
using LinearAlgebra
using LoopVectorization

function profile_ssm_ops()
    model, tok = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B/")
    
    prompt_tokens = Inferno.Tokenizer.encode(tok, "test")
    
    # Warm up
    Inferno.ModelCPU.reset_states_cpu!(model)
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    _ = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
    
    println("=== SSM Layer Operation Breakdown ===")
    println()
    
    # Find an SSM layer
    ssm_layer_idx = findfirst(l -> l.is_ssm, model.layers)
    ssm_layer = model.layers[ssm_layer_idx]
    ssm = ssm_layer.op
    
    println("SSM layer $ssm_layer_idx:")
    println("  in_proj: ", size(ssm.in_proj))
    println("  gate_proj: ", size(ssm.gate_proj))
    println("  ssm_out: ", size(ssm.ssm_out))
    println("  conv_kernel: ", ssm.conv_kernel)
    println("  conv_channels: ", ssm.conv_channels)
    println()
    
    # Benchmark individual operations
    x = model.embed[:, prompt_tokens[end]]
    
    # Warm up
    Inferno.ModelCPU.rmsnorm_cpu!(ssm_layer.norm_buf1, x, ssm_layer.in_norm)
    
    # 1. Input norm
    times_norm = Float64[]
    for i in 1:20
        t0 = time()
        Inferno.ModelCPU.rmsnorm_cpu!(ssm_layer.norm_buf1, x, ssm_layer.in_norm)
        t1 = time()
        push!(times_norm, t1 - t0)
    end
    println("Input norm: $(round(sum(times_norm)/length(times_norm)*1000, digits=3)) ms")
    
    # 2. in_proj matmul
    h_norm = ssm_layer.norm_buf1
    qkv = similar(ssm.qkv_buf)
    times_in_proj = Float64[]
    for i in 1:20
        t0 = time()
        mul!(qkv, ssm.in_proj, h_norm)
        t1 = time()
        push!(times_in_proj, t1 - t0)
    end
    println("in_proj matmul: $(round(sum(times_in_proj)/length(times_in_proj)*1000, digits=3)) ms")
    
    # 3. gate_proj matmul
    z = similar(ssm.z_buf)
    times_gate = Float64[]
    for i in 1:20
        t0 = time()
        mul!(z, ssm.gate_proj, x)
        t1 = time()
        push!(times_gate, t1 - t0)
    end
    println("gate_proj matmul: $(round(sum(times_gate)/length(times_gate)*1000, digits=3)) ms")
    
    # 4. Output projection
    y_all = similar(ssm.y_all_buf)
    out = similar(ssm.out_buf)
    times_out = Float64[]
    for i in 1:20
        t0 = time()
        mul!(out, ssm.ssm_out, y_all)
        t1 = time()
        push!(times_out, t1 - t0)
    end
    println("ssm_out matmul: $(round(sum(times_out)/length(times_out)*1000, digits=3)) ms")
    
    # Total
    println()
    total_matmuls = sum(times_in_proj) + sum(times_gate) + sum(times_out)
    println("Total matmuls: $(round(total_matmuls/length(times_in_proj)*1000, digits=2)) ms")
end

profile_ssm_ops()
