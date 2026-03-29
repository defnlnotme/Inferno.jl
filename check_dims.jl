using Inferno

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check dimensions
println("Config:")
println("  ssm_inner_size (d_inner): ", model.config.ssm_inner_size)
println("  ssm_time_step_rank (num_v_heads): ", model.config.ssm_time_step_rank)
println("  ssm_group_count (num_k_heads): ", model.config.ssm_group_count)
println("  ssm_state_size (head_k_dim): ", model.config.ssm_state_size)

d_inner = model.config.ssm_inner_size
num_v_heads = model.config.ssm_time_step_rank
num_k_heads = model.config.ssm_group_count
head_v_dim = d_inner ÷ num_v_heads
head_k_dim = model.config.ssm_state_size

println("\nDerived dimensions:")
println("  head_v_dim: ", head_v_dim)
println("  head_k_dim: ", head_k_dim)
println("  y_all_cpu size: ", d_inner)
println("  Reshape to (head_v_dim, num_k_heads): (", head_v_dim, ", ", num_k_heads, ") = ", head_v_dim * num_k_heads)
println("  Expected reshape size: ", head_v_dim * num_v_heads)

if num_v_heads != num_k_heads
    println("\n⚠ MISMATCH: num_v_heads != num_k_heads")
    println("  y_all_cpu has size for num_v_heads, but reshape uses num_k_heads!")
end
