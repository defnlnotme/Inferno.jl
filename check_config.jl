using Inferno

model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("=== Model Config ===")
println("hidden_size: ", model.config.hidden_size)
println("num_attention_heads: ", model.config.num_attention_heads)
println("num_key_value_heads: ", model.config.num_key_value_heads)
println("head_dim: ", model.config.head_dim)
println("intermediate_size: ", model.config.intermediate_size)

# Calculate expected dimensions
hidden = model.config.hidden_size
n_heads = model.config.num_attention_heads
n_kv_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim

println("\n=== Expected Attention Weights ===")
println("wq: (", n_heads * head_dim, ", ", hidden, ")")
println("wk: (", n_kv_heads * head_dim, ", ", hidden, ")")
println("wv: (", n_kv_heads * head_dim, ", ", hidden, ")")
println("wo: (", hidden, ", ", n_heads * head_dim, ")")
