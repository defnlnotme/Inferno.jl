# Qwen3.5 CPU Inference Debug Plan

## Goal
Fix CPU inference by tracking line-by-line the llama.cpp implementation.

## Key Files to Compare
1. **llama.cpp**: `src/models/qwen35.cpp` - Model building
2. **llama.cpp**: `src/models/delta-net-base.cpp` - SSM implementation
3. **Inferno**: `src/ModelCPU.jl` - Our CPU implementation

## Components to Check
1. Weight loading and transposition
2. Embedding lookup
3. RMSNorm implementation
4. SSM (GatedDeltaNet) implementation
5. Attention implementation
6. MLP implementation
7. Final norm and LM head

## Tracking Format
For each component, track:
- Variable names
- Types (Float32/Float16, dimensions)
- Operations (matmul, norm, activation)
- Weight shapes and transpositions
