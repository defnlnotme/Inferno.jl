Use the generate_reference.py script as a reference CPU inference implementation.
We are working on Qwen3.5

## Model Implementation Reference

When implementing model support, ALWAYS check the reference implementation in HuggingFace transformers first. For example, for Qwen3.5:

```
~/.local/lib/python$PYTHON_VERSION/site-packages/transformers/models/qwen3_5/
```

Key files to examine:
- `modeling_qwen3_5.py` - Main model architecture, forward pass logic
- `configuration_qwen3_5.py` - Model configuration and hyperparameters
- `tokenization_qwen3_5.py` - Tokenizer implementation

The transformers implementation is the ground truth for:
1. Layer normalization (RMSNorm vs LayerNorm, +1 bias convention)
2. Attention patterns (RoPE, sliding window, GQA)
3. MLP structure (gate * up, then down)
4. Any architecture-specific quirks (SSM for Mamba-like models, etc.)

Always compare your Julia implementation against the transformers reference, not llama.cpp, as llama.cpp may have its own interpretation of the architecture.

## Julia Debugging with DaemonMode.jl

Use DaemonMode.jl to speed up Julia debugging - it avoids REPL startup overhead for each test run.

**Setup:**
```bash
# Terminal 1: Start the daemon
julia --project=. -e 'using DaemonMode; run_daemon()'

# Terminal 2: Run tests via daemon (much faster)
julia --project=. -e 'using DaemonMode; run_job("tests/your_test.jl")'
```

**Benefits:**
- First run compiles packages once, subsequent runs are near-instant
- No Julia startup overhead between test iterations
- Particularly useful when iterating on SSM/attention implementations

**Alternative: Use the REPL directly**
```julia
# In a Julia REPL (julia --project=.)
using DaemonMode
run_daemon()  # Start daemon in background
# In another terminal or same session:
run_job("tests/your_test.jl")
```

## Reference Test Suite

A comprehensive test suite comparing Julia operations with HuggingFace transformers side-by-side is located at:
- `tests/reference/test_ssm_against_hf.py` - SSM operations comparison
- `tests/reference/test_attention_against_hf.py` - Attention operations comparison
- `tests/reference/test_full_model_against_hf.py` - End-to-end model comparison

Run these tests to verify that Julia implementation matches the reference.

## CRITICAL: Multi-Token Generation Verification

**NEVER claim inference is working based on single-token tests!**

When verifying LLM inference correctness, you MUST:
1. Generate at least 128 tokens for verification
2. Check that output is COHERENT TEXT (not repetition, garbage, or punctuation loops)
3. Run these specific test prompts:
   - "What is 2 + 2 ?" - Should answer "4" with explanation
   - "The capital of France" - Should complete with "is Paris" or similar

Single-token top-k matching proves NOTHING about correctness. The model could produce:
- Repetitive punctuation (".\n.\n.\n...")
- Garbage text
- Coherent start then degradation

Only multi-token generation verification counts as success.
