---
trigger: always_on
---

When testing use qwen3.5 0.8B model in tests/models with UD-IQ2_XXS quantization format.
Prefer GPU.1 for testing (not the first GPU)

Always use one liner `echo "What is 2+2?" | julia --project=. examples/simple_prompt.jl` for testing if the (default) model works.