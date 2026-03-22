#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "what is 2 + 2?"


def choose_device() -> tuple[str, torch.dtype]:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Qwen3.5 inference with transformers and PyTorch.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--use-chat-template", action="store_true")
    args = parser.parse_args()

    device, dtype = choose_device()

    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"DType: {dtype}")
    print(f"Prompt: {args.prompt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)

    if args.use_chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        model_input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(model_input_text, return_tensors="pt", add_special_tokens=False)
    else:
        model_input_text = args.prompt
        encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = int(encoded["input_ids"].shape[1])
    generated_ids = output_ids[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)

    print("Input token ids:", encoded["input_ids"][0].tolist())
    print("Generated token ids:", generated_ids)
    print("Generated text:", generated_text)
    print("Full decoded output:", full_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
