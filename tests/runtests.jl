using Test
using Inferno

const MODEL_PATH = get(ENV, "INFERNO_MODEL_PATH", "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
const MODEL_EXISTS = isfile(MODEL_PATH)

@info "Running Inferno.jl Test Suite"
@info "Model path: $MODEL_PATH"
@info "Model exists: $MODEL_EXISTS"

# ===================
# Unit Tests
# ===================
include("unit/test_gguf.jl")
include("unit/test_tokenizer.jl")
include("unit/test_engine.jl")
include("unit/test_server_auth.jl")
include("unit/test_inferno_utils.jl")
include("unit/core_components.jl")

# ===================
# Diagnostic Tests (require model)
# ===================
if MODEL_EXISTS
    @info "Running diagnostic tests with model: $MODEL_PATH"
    
    # BFloat16 support check
    include("diagnostics/check_bfloat16.jl")
end

# ===================
# Legacy Tests (standalone scripts)
# ===================
# These are legacy test scripts that can be run manually
# They are not included in automated test runs as they are interactive/debug scripts
# To run legacy tests manually:
#   julia --project tests/legacy/test_forward.jl
#   julia --project tests/legacy/test_greedy.jl
# etc.

# ===================
# Julia vs Python Tests (require model and Python)
# ===================
# Uncomment when Python/huggingface infrastructure is ready
# include("julia_vs_python/test_prefill_comparison.jl")

# ===================
# Integration Tests (require model and server)
# ===================
# Uncomment when server infrastructure is ready
# include("integration/test_server.jl")
# include("integration/test_cpu_gpu_compare.jl")
