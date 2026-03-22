push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Inferno
using oneAPI

model_path = get(ENV, "INFERNO_MODEL", "unsloth/Qwen3.5-0.8B-GGUF")

println("Starting verification of pure Julia loading...")

try
    # Target second GPU if available
    model, tok = load_model(model_path; device=2)
    println("Model loaded successfully on GPU.")

    # Try a simple forward pass
    input_ids = tok.bos_id != -1 ? [tok.bos_id, 100, 200, 300] : [100, 200, 300]
    println("Running forward pass with dummy input...")
    logits = Inferno.Engine.generate_stream(model, tok, "Hello, how are you?")
    println("Forward pass initiated (streaming).")

    println("Verification complete.")
catch e
    println("Verification failed with error:")
    show(stdout, MIME"text/plain"(), e)
    println()
    rethrow(e)
end
