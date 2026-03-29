using Inferno

file = Inferno.GGUF.read_gguf("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check the conv1d tensor shape in GGUF
tensor = file.tensors["blk.0.ssm_conv1d.weight"]
println("GGUF tensor dimensions: ", tensor.dimensions)
println("GGUF tensor shape: ", tensor.shape)

# Load it
conv1d = Inferno.LoaderCPU.extract_tensor_cpu(file, "blk.0.ssm_conv1d.weight")
println("Extracted shape (before transpose): ", size(conv1d))
println("After transpose: ", size(Float32.(conv1d)'))

# Check in model
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
ssm = model.layers[1].op
println("\nIn model:")
println("ssm_conv1d shape: ", size(ssm.ssm_conv1d))
