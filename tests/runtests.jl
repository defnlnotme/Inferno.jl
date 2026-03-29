using Test
using Inferno

const MODEL_PATH = get(ENV, "INFERNO_MODEL_PATH", "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
const MODEL_EXISTS = isfile(MODEL_PATH)

@info "Running Inferno.jl Test Suite"
@info "Model path: $MODEL_PATH"
@info "Model exists: $MODEL_EXISTS"

# Unit tests (these always run)
include("unit/test_gguf.jl")
include("unit/test_tokenizer.jl")
include("unit/test_engine.jl")
include("unit/test_server_auth.jl")
include("unit/test_inferno_utils.jl")
include("unit/core_components.jl")

# Integration tests are skipped for now (require running server)
# Uncomment when server infrastructure is ready
# if MODEL_EXISTS
#     @info "Running integration tests with model: $MODEL_PATH"
#     include("integration/test_server.jl")
# end
