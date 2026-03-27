using Inferno

# Load model
model_path = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
model, file = LoaderCPU.load_model_cpu(model_path)

# Get tokenizer
tokenizer = file.metadata["tokenizer.ggml.tokens"]

# Create decode function
decode_fn = (ids) -> join([replace(tokenizer[i + 1], "Ġ" => " ") for i in ids])
