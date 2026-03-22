#!/usr/bin/env python3
"""Save dequantized reference values as raw binary files."""

import numpy as np, gguf, os

GGUF_PATH = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
OUT_DIR = "examples/dequant_ref"
os.makedirs(OUT_DIR, exist_ok=True)
r = gguf.GGUFReader(GGUF_PATH)

test_cases = {
    "F32": "output_norm.weight",
    "F16": "blk.0.ssm_alpha.weight",
    "Q8_0": "blk.0.ssm_out.weight",
    "Q4_K": "blk.0.ffn_gate.weight",
    "Q5_K": "blk.0.attn_qkv.weight",
    "Q6_K": "token_embd.weight",
    "IQ4_XS": "blk.8.ffn_gate.weight",
}

for qtype, tname in test_cases.items():
    for t in r.tensors:
        if t.name == tname:
            d = gguf.dequantize(t.data, t.tensor_type).astype(np.float64)
            with open(os.path.join(OUT_DIR, f"{qtype}.bin"), "wb") as f:
                np.array([len(d.shape)], dtype=np.int64).tofile(f)
                np.array(d.shape, dtype=np.int64).tofile(f)
                d.ravel(order="C").tofile(f)
            print(f"  {qtype:6s} shape={d.shape} norm={np.linalg.norm(d):.6f}")
            break

print(f"Saved to {OUT_DIR}/")
