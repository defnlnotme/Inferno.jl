using Inferno

# Load model
model_path = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
model, file = LoaderCPU.load_model_cpu(model_path)

# Get tokenizer
tokens_data = file.metadata["tokenizer.ggml.tokens"]

# Create decode function
decode_fn = (ids) -> join([replace(tokens_data[i + 1], "Ġ" => " ") for i in ids])

# Stream generation to stdout
stream_to_stdout_cpu(
    model,
    [562],  # prompt tokens (e.g., " The")
    decode_fn;
    max_tokens=100,
    temperature=0.7f0,
    top_p=0.95f0,
    top_k=40,
    repetition_penalty=1.1f0
)
