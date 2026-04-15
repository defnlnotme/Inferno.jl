# ENV["MODEL_PATH"] = joinpath(@__DIR__, "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
ENV["MODEL_PATH"] = joinpath(@__DIR__, "tests/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")
global const MODEL_PATH = ENV["MODEL_PATH"]
using Inferno
model, bpetok = load_model(MODEL_PATH, backend=:cpu)
stream_to_stdout(model, bpetok, "2 + 2 =" , top_p=0.8, top_k=20, presence_penalty=1.5, repetition_penalty=1.0, temperature=0.2, max_tokens=120, backend=:cpu, show_tps=true);
