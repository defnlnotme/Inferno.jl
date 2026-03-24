# Inferno Test Suite

This directory contains the complete test suite for the Inferno project.

## Quick Start

```bash
cd tests

# Run all tests
./run.sh

# Run only unit tests (fast, no model required)
./run.sh unit

# Run integration tests
./run.sh integration

# Run with coverage tracking
./run.sh coverage
```

## Test Organization

Tests are organized by their dependencies:

| Marker | Description | Dependencies |
|--------|-------------|--------------|
| `[UNIT]` | Fast, isolated tests | None (no model/GPU) |
| `[MODEL]` | Model-dependent tests | GGUF model file |
| `[GPU]` | GPU-dependent tests | oneAPI + Intel GPU |
| `[INTEGRATION]` | Full system tests | HTTP server, model |

## Directory Structure

```
tests/
├── README.md           # This file
├── runtests.jl         # Main test runner (use `julia --project=. runtests.jl`)
├── Project.toml        # Test dependencies
│
├── unit/ # Unit tests for individual components
│ ├── core_components.jl # Core component tests (legacy)
│ ├── test_engine.jl # [UNIT] Engine sampling tests
│ ├── test_tokenizer.jl # [UNIT] Tokenizer edge cases
│ ├── test_gguf.jl # [UNIT] GGUF parsing tests
│ ├── test_inferno_utils.jl # [UNIT] Utility function tests
│ └── test_server_auth.jl # [INTEGRATION] Server auth tests
│
├── integration/        # Integration tests (multi-component, server, etc.)
│   └── test_server.jl
│
├── debugging/          # Debug and diagnostic tests
│   ├── 01_loading.jl
│   ├── 02_tokenization.jl
│   ├── 03_embedding_norms.jl
│   ├── critical_components.jl
│   └── debug_inference.jl
│
├── benchmarks/         # Performance benchmarks
│   └── benchmark_forward.jl
│
├── julia/              # Pure Julia verification tests
│   ├── generate_text.jl
│   ├── shared.jl
│   └── verify_all.jl
│
├── python/             # Python comparison scripts
│   ├── common.py
│   ├── compare_generation.py
│   ├── generate_all.py
│   └── generate_text.py
│
└── models/             # Local model files (git-ignored)
    ├── .cache/
    ├── Qwen3.5-0.8B/
    └── Qwen3.5-0.8B-GGUF/
```

## Test Categories

### Unit Tests (`unit/`)
Fast, isolated tests for individual components:
- GGUF parsing
- Tokenizer
- RMSNorm
- Config extraction
- Dequantization kernels
- Server prompt building

### Integration Tests (`integration/`)
Tests that verify multiple components work together:
- Server endpoints (HTTP API)
- Full model inference pipeline

### Debugging Tests (`debugging/`)
Diagnostic tests for troubleshooting:
- Step-by-step loading verification
- Tokenization debugging
- Embedding norm analysis
- Critical component health checks
- Full inference bisection

### Benchmarks (`benchmarks/`)
Performance measurement scripts:
- Forward pass timing
- Memory usage profiling

### Julia Tests (`julia/`)
Pure Julia implementation verification:
- Reference implementation tests
- Python comparison verification

### Python Scripts (`python/`)
Helper scripts for generating reference outputs:
- Generate all reference data
- Compare Julia vs Python outputs

## Running Tests

### Full Test Suite
```bash
cd tests
julia --project=. runtests.jl
```

### Unit Tests Only
```bash
julia --project=. -e 'using Test; include("unit/core_components.jl")'
```

### Integration Tests
```bash
julia --project=. -e 'using Test; include("integration/test_server.jl")'
```

### Debugging Tests
```bash
julia --project=. debugging/debug_inference.jl
```

### Benchmarks
```bash
julia --project=. benchmarks/benchmark_forward.jl
```

## Environment Variables

- `INFERNO_MODEL`: Path to the model file (default: `unsloth/Qwen3.5-0.8B-GGUF`)

Example:
```bash
export INFERNO_MODEL=/path/to/model.gguf
julia --project=. runtests.jl
```

## Model Files

The `models/` directory contains local model files and is git-ignored. Place your
GGUF model files here for testing:
```
models/Qwen3.5-0.8B-GGUF/your-model.gguf
```
