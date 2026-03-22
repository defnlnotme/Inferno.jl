# Julia vs Python Comparison Tests

This directory contains tests that compare the output of the Julia implementation against the Python (HuggingFace transformers) implementation.

## Tests

### `test_prefill_comparison.jl`

Compares the `prefill_prompt_tokens` function output between Julia and Python to ensure they produce identical results.

**What it tests:**
- Token IDs match exactly between Julia and Python tokenizers
- Logits match within floating-point tolerance

**Note:** Hidden state comparison is not included in the basic test as it requires instrumenting the forward pass. The Python script can be modified to output hidden states if needed.

## Setup

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Model Path

Set the `INFERNO_MODEL` environment variable to point to your model:

```bash
export INFERNO_MODEL=unsloth/Qwen3.5-0.8B-GGUF
# or
export INFERNO_MODEL=/path/to/your/model.gguf
```

## Running the Tests

### Run the full test suite (includes Julia vs Python tests):

```bash
julia --project=tests tests/runtests.jl
```

### Run only the prefill comparison test:

```bash
julia --project=. -e '
using Test
include("tests/julia_vs_python/test_prefill_comparison.jl")
'
```

## Python Script

### `prefill_comparison.py`

This script extracts prefill hidden states and logits using the transformers library. It can be run standalone:

```bash
python prefill_comparison.py <model_path> <prompt> [use_chat_template] [system_prompt]
```

**Arguments:**
- `model_path`: Path to model or HuggingFace model name
- `prompt`: Input prompt (use quotes for multi-word prompts)
- `use_chat_template`: `true`/`false` (default: `true`)
- `system_prompt`: Optional system prompt

**Output:** JSON object with:
- `token_ids`: List of token IDs
- `logits`: Last token's logits

## Tolerance Levels

The tests use the following tolerances for floating-point comparisons:

- **Logits**: `atol=1e-1`, `rtol=1e-1`

These tolerances account for differences in:
- Floating-point arithmetic order
- GPU vs CPU computation
- Different library implementations (oneAPI vs CUDA vs CPU)
- Quantization effects (if using quantized GGUF models in Julia)

## Troubleshooting

### Token IDs don't match

This indicates a tokenizer implementation difference. Check:
- Both implementations use the same tokenizer version
- Chat template is applied consistently
- BOS/EOS tokens are handled the same way

### Logits don't match

Large differences (>10%) may indicate:
- Different model weights being loaded
- Quantization differences (Julia uses GGUF quantized models)
- Architecture implementation bugs

Small differences (<10%) are expected due to:
- Floating-point precision
- Different computation order
- Quantization effects in GGUF models

## Model Compatibility

**Important:** The Julia implementation uses GGUF quantized models, while Python uses full precision FP16 models from HuggingFace. Some differences in logits are expected due to quantization.

For best comparison results:
1. Use a low-quantization model (e.g., Q8_0 or F16 GGUF)
2. Or use the Python script with a quantized model if available
