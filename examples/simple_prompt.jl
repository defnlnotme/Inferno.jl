using Inferno

# Path to the model (now relative to the examples/ folder)
model_path = joinpath(@__DIR__, "..", "tests", "models", "Qwen3.5-0.8B-UD-IQ2_XXS.gguf")

# 1. Load the model and tokenizer
model, tok = Inferno.load_model(model_path)

# 2. Define your prompt
println("-"^40)
print("Enter prompt: ")
prompt = readline()
if isempty(prompt)
    prompt = "The capital of France is"
end

# 3. Generate and print (streaming)
println("\nGenerating response...")
println("-"^40)
print("Response: ")

# generate_stream yields one string token (decoded) at a time
for token in Inferno.generate_stream(model, tok, prompt; max_tokens=20, temperature=0.1f0)
    print(token)
    flush(stdout)
end
println("\n" * "-"^40)
