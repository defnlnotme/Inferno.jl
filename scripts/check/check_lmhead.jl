using Inferno

model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

println("lm_head[1:5, 1:5]:")
println(model.lm_head[1:5, 1:5])
println("\nembed'[1:5, 1:5]:")
println(model.embed'[1:5, 1:5])

# Check max difference
diff = maximum(abs.(model.lm_head - model.embed'))
println("\nMax diff between lm_head and embed': ", diff)
