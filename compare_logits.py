from llama_cpp import Llama
import numpy as np

llm = Llama(
    model_path="tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
    n_ctx=512,
    logits=True,
    verbose=False
)

prompt = "The capital of France is"
output = llm(prompt, max_tokens=0, echo=False)

logits = np.array(output['choices'][0]['logits'][0])

# Top 10 tokens
top_indices = np.argsort(logits)[::-1][:10]
print("Top 10 tokens after prompt (llama.cpp):")
for idx in top_indices:
    token = llm.detokenize([int(idx)]).decode('utf-8', errors='replace')
    print(f"  {idx}: \"{token}\" (logit: {logits[idx]:.3f})")

# Specific tokens
print("\nSpecific tokens:")
test_tokens = [272, 221, 59, 3368, 7048, 61]
for t in test_tokens:
    token = llm.detokenize([t]).decode('utf-8', errors='replace')
    print(f"  Token {t} (\"{token}\"): logit {logits[t]:.3f}")
