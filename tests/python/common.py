"""Shared utilities for Python test scripts. Reuses gguf library for weight loading."""

import numpy as np
import gguf

GGUF_PATH = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"


def load_gguf():
    return gguf.GGUFReader(GGUF_PATH)


def get_config(r):
    md = r.fields

    def g(key, default=None):
        if key in md:
            f = md[key]
            if f.data is not None and len(f.data) > 0 and f.parts is not None:
                val = f.parts[f.data[0]]
                if hasattr(val, "item"):
                    return val.item()
                return val
        return default

    return {
        "vocab_size": int(g("qwen35.vocab_size", 248320)),
        "hidden_size": int(g("qwen35.embedding_length", 1024)),
        "intermediate_size": int(g("qwen35.feed_forward_length", 3584)),
        "num_hidden_layers": int(g("qwen35.block_count", 24)),
        "num_attention_heads": int(g("qwen35.attention.head_count", 8)),
        "num_key_value_heads": int(g("qwen35.attention.head_count_kv", 2)),
        "head_dim": int(g("qwen35.attention.key_length", 256)),
        "rms_norm_eps": float(g("qwen35.attention.layer_norm_rms_epsilon", 1e-6)),
        "rope_theta": float(g("qwen35.rope.freq_base", 1e7)),
        "max_pos": min(4096, int(g("qwen35.context_length", 262144))),
        "full_attention_interval": int(g("qwen35.full_attention_interval", 4)),
        "ssm_inner_size": int(g("qwen35.ssm.inner_size", 2048)),
        "ssm_state_size": int(g("qwen35.ssm.state_size", 128)),
        "ssm_group_count": int(g("qwen35.ssm.group_count", 16)),
        "ssm_time_step_rank": int(g("qwen35.ssm.time_step_rank", 16)),
        "ssm_conv_kernel": int(g("qwen35.ssm.conv_kernel", 4)),
    }


def get_tensor(r, name):
    for t in r.tensors:
        if t.name == name:
            return t
    return None


def load_weight(r, name):
    """Load dequantized weight. Returns (out_features, in_features) for 2D, (n,) for 1D."""
    t = get_tensor(r, name)
    if t is None:
        raise KeyError(f"Tensor not found: {name}")
    if t.tensor_type == 0:
        return np.array(t.data, dtype=np.float32)
    elif t.tensor_type == 1:
        return np.array(t.data, dtype=np.float16).astype(np.float32)
    return gguf.dequantize(t.data, t.tensor_type).astype(np.float32)


def rmsnorm(x, weight, eps):
    ss = np.sum(x.astype(np.float64) ** 2)
    scale = 1.0 / np.sqrt(ss / len(x) + eps)
    return (x * scale * weight).astype(np.float32)


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def assert_close(a, b, name, rtol=1e-3):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-10)
    cos = np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    status = "PASS" if rel < rtol and cos > 0.9999 else "FAIL"
    print(f"  {status}  {name:40s} rel_err={rel:.2e} cos_sim={cos:.8f}")
    assert rel < rtol and cos > 0.9999, f"{name} failed: rel_err={rel}, cos_sim={cos}"


TOKEN_ID = 151646  # 0-indexed
