"""
Python script to extract prefill hidden states using transformers library.
This is used to compare with Julia's prefill_prompt_tokens output.
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_prefill_hidden_states(
    model_path: str,
    prompt: str,
    use_chat_template: bool = True,
    system_prompt: str = None,
):
    """
    Get the hidden states after prefilling the prompt.

    Args:
        model_path: Path to the model or HuggingFace model name
        prompt: The input prompt
        use_chat_template: Whether to apply chat template
        system_prompt: Optional system prompt for chat template

    Returns:
        Dictionary containing:
        - token_ids: List of token IDs
        - hidden_state: Last token's hidden state as list
        - logits: Last token's logits as list
    """
    # Load model and tokenizer
    print(f"Loading model from: {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # Apply chat template if requested
    if use_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Add generation prompt
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = prompt

    print(f"Rendered prompt: {rendered[:100]}...", file=sys.stderr)

    # Tokenize
    inputs = tokenizer.encode(rendered, return_tensors="pt").to(model.device)
    token_ids = inputs.tolist()[0]
    print(f"Token IDs: {token_ids}", file=sys.stderr)
    print(f"Number of tokens: {len(token_ids)}", file=sys.stderr)

    # Forward pass through the model to get hidden states
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits[0, -1, :]  # Get last token's logits

    # Convert to lists for JSON serialization
    result = {"token_ids": token_ids, "logits": logits.cpu().tolist()}

    return result


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python prefill_comparison.py <model_path> <prompt> [use_chat_template] [system_prompt]",
            file=sys.stderr,
        )
        print("  model_path: Path to model or HuggingFace model name", file=sys.stderr)
        print(
            "  prompt: Input prompt (use quotes for multi-word prompts)",
            file=sys.stderr,
        )
        print("  use_chat_template: true/false (default: true)", file=sys.stderr)
        print("  system_prompt: Optional system prompt", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2]
    use_chat_template = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True
    system_prompt = sys.argv[4] if len(sys.argv) > 4 else None

    result = get_prefill_hidden_states(
        model_path, prompt, use_chat_template, system_prompt
    )

    # Output as JSON
    print(json.dumps(result))


if __name__ == "__main__":
    main()
