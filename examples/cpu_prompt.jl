using Inferno

# Load model
model_path = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
model, tokenizer = LoaderCPU.load_model_cpu(model_path)

# Use BPETokenizer's decode method
decode_fn = (ids) -> Tokenizer.decode(tokenizer, ids)
