#!/usr/bin/env python3
"""
Generate text from Qwen3.5 using forward_ref.py's forward pass with actual prompt.
This generates text by running the forward pass step by step and sampling tokens.
"""

import sys
import os

sys.path.insert(0, "examples/checks")
sys.path.insert(0, "tests/python")
import argparse
import time
import numpy as np

from common import GGUF_PATH
from forward_ref import forward_pass, load_gguf, get_config


def greedy_sample(logits):
    """Simple greedy sampling - pick the token with highest logit."""
    return int(np.argmax(logits))


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with forward_ref forward pass"
    )
    parser.add_argument("prompt", type=str, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens")
    parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print("--- Model Info ---")
    print(f"Model path: {GGUF_PATH}")
    print()
    print("--- Generation Parameters ---")
    print(f"Prompt: {args.prompt!r}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Load tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "unsloth/Qwen3.5-0.8B",
        trust_remote_code=True,
    )
    print(f"Tokenizer loaded. Vocab size: {tok.vocab_size}")

    # Apply chat template
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tok.encode(text, add_special_tokens=False)
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:10]}...")
    print()

    # Load model and run forward pass
    print("Loading model...")
    load_start = time.perf_counter()
    r = load_gguf(GGUF_PATH)
    cfg = get_config(r)
    print(f"Config: hidden={cfg['hidden_size']}, layers={cfg['num_hidden_layers']}")
    print(f"Model loaded in {time.perf_counter() - load_start:.2f}s")
    print()

    # Run forward pass for each token in prompt
    print("--- Generating ---")
    gen_start = time.perf_counter()
    generated_ids = []

    for step in range(args.max_tokens):
        # Get position
        if step < len(prompt_tokens):
            token_id = prompt_tokens[step]
        else:
            # Autoregressive generation
            if not generated_ids:
                break  # No tokens generated, something wrong
            token_id = generated_ids[-1]

        pos = step
        outputs = forward_pass(r, token_id, cfg)

        logits = outputs["logits"]

        if step >= len(prompt_tokens):
            # After prompt, sample next token
            next_token = greedy_sample(logits)
            generated_ids.append(next_token)

            if next_token == tok.eos_token_id:
                print(f"EOS token {next_token} reached")
                break

            # Decode and print
            token_str = tok.decode([next_token], skip_special_tokens=True)
            print(f"Token {step}: {next_token} -> {repr(token_str)}")

    gen_elapsed = time.perf_counter() - gen_start

    # Decode full response
    if generated_ids:
        response_text = tok.decode(generated_ids, skip_special_tokens=True)
    else:
        response_text = ""

    print()
    print("--- Results ---")
    print(f"Generated text: {response_text!r}")
    print(f"Token IDs: {generated_ids}")
    print(f"Tokens generated: {len(generated_ids)}")
    print(f"Elapsed: {gen_elapsed:.3f}s")
    if gen_elapsed > 0:
        print(f"Throughput: {len(generated_ids) / gen_elapsed:.1f} tok/s")

    if args.output_json:
        import json

        print(
            json.dumps(
                {
                    "text": response_text,
                    "token_ids": generated_ids,
                    "tokens_generated": len(generated_ids),
                    "elapsed": gen_elapsed,
                    "tok_per_sec": len(generated_ids) / gen_elapsed
                    if gen_elapsed > 0
                    else 0,
                }
            )
        )


if __name__ == "__main__":
    main()
