#!/usr/bin/env python3
"""
Python reference forward pass for Qwen3.5 hybrid model.
Reads the GGUF file directly, implements the EXACT same computation
as the Julia inference.jl forward pass, and captures hidden states
at each layer boundary for comparison.

Usage:
    python3 forward_ref.py [--token-id N] [--layers 24]

Output:
    Writes numpy arrays to examples/reference_outputs/
"""

import numpy as np
import gguf
import os
import sys
import time

# ── Model config from GGUF ──────────────────────────────────────────────

GGUF_PATH = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
OUTPUT_DIR = "examples/reference_outputs"

# Default test: single token id
DEFAULT_TOKEN_ID = 151646  # <|im_start|> in Qwen tokenizer

# ── Helpers ─────────────────────────────────────────────────────────────


def load_gguf(path):
    return gguf.GGUFReader(path)


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
        "full_attn_interval": int(g("qwen35.full_attention_interval", 4)),
        "ssm_inner_size": int(g("qwen35.ssm.inner_size", 2048)),
        "ssm_state_size": int(g("qwen35.ssm.state_size", 128)),
        "ssm_group_count": int(g("qwen35.ssm.group_count", 16)),
        "ssm_time_step_rank": int(g("qwen35.ssm.time_step_rank", 16)),
        "ssm_conv_kernel": int(g("qwen35.ssm.conv_kernel", 4)),
    }


def find_tensor(r, name):
    for t in r.tensors:
        if t.name == name:
            return t
    return None


def get_weight(r, name):
    """Dequantize a tensor. Note: GGUF declared shape is transposed vs dequantized shape."""
    t = find_tensor(r, name)
    if t is None:
        raise KeyError(f"Tensor not found: {name}")
    if t.tensor_type == 0:  # F32
        return np.array(t.data, dtype=np.float32)
    elif t.tensor_type == 1:  # F16
        return np.array(t.data, dtype=np.float16).astype(np.float32)
    else:
        return gguf.dequantize(t.data, t.tensor_type).astype(np.float32)


def get_vector(r, name):
    """Get a 1D tensor (bias, norm weight, etc.)"""
    return get_weight(r, name).flatten()


# ── Operations ──────────────────────────────────────────────────────────


def rmsnorm(x, weight, eps):
    """RMSNorm: x * (1 / sqrt(mean(x^2) + eps)) * weight"""
    # Julia: ss = sum(abs2.(x)), m = ss / N, scale = 1/sqrt(m + eps)
    ss = np.sum(x.astype(np.float64) ** 2)  # sum of squares, upcast for precision
    mean_sq = ss / x.shape[0]
    scale = 1.0 / np.sqrt(mean_sq + eps)
    return (x * scale * weight).astype(np.float32)


def silu(x):
    """SiLU / Swish: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


# ── RoPE cache ──────────────────────────────────────────────────────────


def build_rope_cache(cfg):
    head_dim = cfg["head_dim"]
    rope_dim = head_dim  # qk_rope_head_dim is 0 for this model
    rope_pairs = rope_dim // 2
    theta = cfg["rope_theta"]
    max_pos = cfg["max_pos"]

    sin_cache = np.zeros((rope_pairs, max_pos), dtype=np.float64)
    cos_cache = np.zeros((rope_pairs, max_pos), dtype=np.float64)

    for pair_idx in range(rope_pairs):
        i = 2 * pair_idx + 1  # 1-indexed: 2*pair_idx-1+1 = 2*pair_idx+1
        freq = 1.0 / (theta ** ((i - 1) / rope_dim))
        for pos in range(max_pos):
            sin_cache[pair_idx, pos] = np.sin(pos * freq)
            cos_cache[pair_idx, pos] = np.cos(pos * freq)

    return sin_cache.astype(np.float32), cos_cache.astype(np.float32), rope_dim


def apply_rope(q_heads, k_heads, pos, sin_cache, cos_cache, rope_dim):
    """
    Apply RoPE to Q and K heads.
    q_heads, k_heads: (head_dim, n_heads) column-major
    Julia code: q_odd = q[1:2:rope_dim], q_even = q[2:2:rope_dim]
                q_odd = tmp * cos - q_even * sin
                q_even = tmp * sin + q_even * cos
    """
    rope_pairs = rope_dim // 2
    sin_vals = sin_cache[:, pos]  # (rope_pairs,)
    cos_vals = cos_cache[:, pos]  # (rope_pairs,)

    for qk in [q_heads, k_heads]:
        # Interleaved pairs: (0,1), (2,3), ..., (rope_dim-2, rope_dim-1)
        odd = qk[0:rope_dim:2, :].copy()  # indices 0, 2, 4, ...
        even = qk[1:rope_dim:2, :].copy()  # indices 1, 3, 5, ...

        # Wait, Julia is 1-indexed: 1:2:rope_dim gives 1,3,5,... and 2:2:rope_dim gives 2,4,6,...
        # In 0-indexed: 0:2:rope_dim gives 0,2,4,... and 1:2:rope_dim gives 1,3,5,...
        # So q_odd = q[0::2], q_even = q[1::2]
        pass  # let me redo this

    # Correct approach:
    for qk in [q_heads, k_heads]:
        odd_idx = np.arange(0, rope_dim, 2)
        even_idx = np.arange(1, rope_dim, 2)
        tmp = qk[odd_idx, :].copy()
        qk[odd_idx, :] = (
            tmp * cos_vals[:, np.newaxis] - qk[even_idx, :] * sin_vals[:, np.newaxis]
        )
        qk[even_idx, :] = (
            tmp * sin_vals[:, np.newaxis] + qk[even_idx, :] * cos_vals[:, np.newaxis]
        )


# ── SSM block ───────────────────────────────────────────────────────────


class SSMState:
    def __init__(self, cfg):
        inner = cfg["ssm_inner_size"]
        state_size = cfg["ssm_state_size"]
        groups = cfg["ssm_group_count"]
        head_v_dim = inner // cfg["ssm_time_step_rank"]
        conv_kernel = cfg["ssm_conv_kernel"]
        conv_channels = 2 * groups * state_size + inner  # = 6144

        self.conv = np.zeros((conv_channels, conv_kernel), dtype=np.float32)
        self.h = np.zeros((head_v_dim, state_size, groups), dtype=np.float64)


def ssm_process(block, x_norm, state, cfg, weights):
    """Exact mirror of Julia process_branch!(Val(:ssm), ...)"""
    inner_size = cfg["ssm_inner_size"]
    state_size = cfg["ssm_state_size"]
    groups = cfg["ssm_group_count"]
    head_v_dim = inner_size // cfg["ssm_time_step_rank"]
    conv_channels = 2 * groups * state_size + inner_size
    eps = cfg["rms_norm_eps"]

    # Step 1: QKV and gate projections
    # attn_qkv weight dequantized shape: (conv_channels, hidden_size)
    qkv_proj = weights["attn_qkv"] @ x_norm  # (conv_channels,)
    z_buf = weights["attn_gate"] @ x_norm  # (inner_size,)

    # Step 2: 1D convolution (ring buffer)
    K = cfg["ssm_conv_kernel"]
    if K > 1:
        state.conv[:, :-1] = state.conv[:, 1:]
    state.conv[:, -1] = qkv_proj

    x_conv = np.zeros(conv_channels, dtype=np.float32)
    for k in range(K):
        x_conv += state.conv[:, k] * weights["conv1d"][:, k]

    # Step 3: SiLU
    x_conv = silu(x_conv)

    # Step 4: Split into Q, K, V
    qk_size = state_size * groups
    q_all = x_conv[:qk_size].reshape(groups, state_size)  # (groups, state_size)
    k_all = x_conv[qk_size : 2 * qk_size].reshape(
        groups, state_size
    )  # (groups, state_size)
    v_all = x_conv[2 * qk_size : 2 * qk_size + inner_size].reshape(
        groups, head_v_dim
    )  # (groups, head_v_dim)

    # Step 5: Alpha/beta projections (CPU Float32 in Julia)
    x_norm32 = x_norm.astype(np.float64)
    alpha_proj = weights["alpha"] @ x_norm32  # (groups,)
    beta_proj = weights["beta"] @ x_norm32  # (groups,)

    dt_bias = weights["dt_bias"]
    ssm_a = weights["ssm_a"]

    y_all = np.zeros(inner_size, dtype=np.float64)

    # Step 6: Per-group SSM recurrence
    for g in range(groups):
        qg = q_all[g, :].astype(np.float64)
        kg = k_all[g, :].astype(np.float64)
        vg = v_all[g, :].astype(np.float64)

        # L2 normalization of Q and K (with eps)
        q_norm_val = np.sqrt(np.sum(qg**2) + eps)
        k_norm_val = np.sqrt(np.sum(kg**2) + eps)
        q_norm = qg / q_norm_val
        k_norm = kg / k_norm_val

        # Decay gate and beta gate
        # Julia: dg = exp(log(1.0 + exp(alpha_proj[g] + dt_bias[g])) * ssm_a[g])
        # Julia: bg = 1.0 / (1.0 + exp(-beta_proj[g]))
        softplus_arg = alpha_proj[g] + dt_bias[g]
        # softplus(x) = log(1 + exp(x))
        dg = np.exp(np.log1p(np.exp(softplus_arg)) * ssm_a[g])
        bg = 1.0 / (1.0 + np.exp(-beta_proj[g]))

        # State decay
        state.h[:, :, g] *= dg

        # Gated Delta Rule update
        # tmp_head = state * k_norm  (matvec)
        # update = bg * (v - tmp_head)
        # state += outer(update, k_norm)
        sk = state.h[:, :, g] @ k_norm  # (head_v_dim,)
        update_val = bg * (vg - sk)
        state.h[:, :, g] += np.outer(update_val, k_norm)

        # Output via state * q_norm
        yg = state.h[:, :, g] @ q_norm  # (head_v_dim,)
        y_all[g * head_v_dim : (g + 1) * head_v_dim] = yg

    # Step 7: Output normalization
    y_reshaped = y_all.reshape(groups, head_v_dim)
    y_normed = np.zeros_like(y_all)
    for g in range(groups):
        y_normed[g * head_v_dim : (g + 1) * head_v_dim] = rmsnorm(
            y_reshaped[g, :], weights["ssm_norm"], eps
        )

    # Step 8: SiLU gate
    z_gated = silu(z_buf)
    y_normed *= z_gated

    # Step 9: Output projection
    branch_out = weights["ssm_out"] @ y_normed
    return branch_out


# ── Attention block ─────────────────────────────────────────────────────


class KVCache:
    def __init__(self, cfg):
        head_dim = cfg["head_dim"]
        n_kv = cfg["num_key_value_heads"]
        max_pos = cfg["max_pos"]
        self.k = np.zeros((n_kv, head_dim, max_pos), dtype=np.float32)
        self.v = np.zeros((n_kv, head_dim, max_pos), dtype=np.float32)


def attn_process(
    block, x_norm, kv_cache, pos, cfg, weights, sin_cache, cos_cache, rope_dim
):
    """Exact mirror of Julia process_branch!(Val(:attn), ...)"""
    head_dim = cfg["head_dim"]
    n_heads_q = cfg["num_attention_heads"]
    n_heads_kv = cfg["num_key_value_heads"]
    eps = cfg["rms_norm_eps"]

    # Step 1: Q, K, V projections
    # attn_q shape in dequantized: (n_heads_q * head_dim * 2, hidden)
    # because q_all_buf packs Q + gate
    q_all_buf = weights["attn_q"] @ x_norm  # (n_heads_q * head_dim * 2,)
    k_buf = weights["attn_k"] @ x_norm  # (n_heads_kv * head_dim,)
    v_buf = weights["attn_v"] @ x_norm  # (n_heads_kv * head_dim,)

    # Step 2: Split Q and gate
    q_size = n_heads_q * head_dim
    q = q_all_buf[:q_size].copy()
    attn_gate = q_all_buf[q_size : 2 * q_size].copy()

    # Step 3: Reshape to (head_dim, n_heads) - column-major in Julia
    # Julia: reshape(q, head_dim, n_heads_q) => our (n_heads_q, head_dim) transposed
    # In numpy with row-major: we keep (n_heads_q, head_dim) and transpose when needed
    q_heads = q.reshape(n_heads_q, head_dim)  # (n_heads_q, head_dim)
    k_heads = k_buf.reshape(n_heads_kv, head_dim)  # (n_heads_kv, head_dim)
    v_heads = v_buf.reshape(n_heads_kv, head_dim)  # (n_heads_kv, head_dim)

    # Step 4: Q and K RMS normalization
    for h in range(n_heads_q):
        q_heads[h, :] = rmsnorm(q_heads[h, :], weights["attn_q_norm"], eps)
    for h in range(n_heads_kv):
        k_heads[h, :] = rmsnorm(k_heads[h, :], weights["attn_k_norm"], eps)

    # Step 5: SiLU gate * Q
    attn_gate_reshaped = attn_gate.reshape(n_heads_q, head_dim)
    attn_gate_gated = silu(attn_gate_reshaped)
    q_heads *= attn_gate_gated

    # Step 6: RoPE (applied to first rope_dim elements of each head)
    # Julia uses (head_dim, n_heads) column-major, we use (n_heads, head_dim) row-major
    # RoPE operates on interleaved pairs: (0,1), (2,3), ... (rope_dim-2, rope_dim-1)
    for h in range(n_heads_q):
        for pair in range(rope_dim // 2):
            i0 = 2 * pair
            i1 = 2 * pair + 1
            s = sin_cache[pair, pos]
            c = cos_cache[pair, pos]
            q0 = q_heads[h, i0]
            q1 = q_heads[h, i1]
            q_heads[h, i0] = q0 * c - q1 * s
            q_heads[h, i1] = q0 * s + q1 * c

    for h in range(n_heads_kv):
        for pair in range(rope_dim // 2):
            i0 = 2 * pair
            i1 = 2 * pair + 1
            s = sin_cache[pair, pos]
            c = cos_cache[pair, pos]
            k0 = k_heads[h, i0]
            k1 = k_heads[h, i1]
            k_heads[h, i0] = k0 * c - k1 * s
            k_heads[h, i1] = k0 * s + k1 * c

    # Step 7: Update KV cache
    kv_cache.k[:, :, pos] = k_heads
    kv_cache.v[:, :, pos] = v_heads

    # Step 8: GQA attention
    attn_out = np.zeros((n_heads_q, head_dim), dtype=np.float64)
    scale = 1.0 / np.sqrt(head_dim)
    gqa_ratio = n_heads_q // n_heads_kv

    for h in range(n_heads_q):
        kv_h = h // gqa_ratio
        K_past = kv_cache.k[kv_h, :, : pos + 1].T  # (pos+1, head_dim)
        V_past = kv_cache.v[kv_h, :, : pos + 1].T  # (pos+1, head_dim)
        q_h = q_heads[h, :].astype(np.float64)

        scores = K_past.astype(np.float64) @ q_h  # (pos+1,)
        scores *= scale

        mx = np.max(scores)
        scores = np.exp(scores - mx)
        scores /= np.sum(scores)

        attn_out[h, :] = scores @ V_past.astype(np.float64)

    # Step 9: Output projection
    attn_out_flat = attn_out.flatten()
    branch_out = weights["attn_output"] @ attn_out_flat.astype(np.float32)
    return branch_out


# ── MLP block ───────────────────────────────────────────────────────────


def mlp_process(x_norm, cfg, weights):
    """Exact mirror of Julia process_mlp!(...)"""
    gate = weights["ffn_gate"] @ x_norm
    up = weights["ffn_up"] @ x_norm
    gate = silu(gate)
    gate *= up
    branch_out = weights["ffn_down"] @ gate
    return branch_out


# ── Main forward pass ───────────────────────────────────────────────────


def forward_pass(r, token_id, cfg):
    """
    Run the full forward pass and capture hidden states at each stage.

    Returns a dict of named numpy arrays:
        "embed"        - after embedding lookup
        "layer{i}_post_branch"  - after first residual (SSM/Attn branch)
        "layer{i}_post_mlp"     - after second residual (MLP branch)
        "final_norm"   - after final RMSNorm
        "logits"       - final logits (full vocab)
    """
    print(f"Loading weights from GGUF...")
    t0 = time.time()

    # Load all weights
    embed = get_weight(r, "token_embd.weight")  # (vocab, hidden)
    output_norm = get_vector(r, "output_norm.weight")

    # Check for separate output weights
    out_w = find_tensor(r, "output.weight")
    lm_head = get_weight(r, "output.weight") if out_w is not None else embed

    blocks = []
    num_layers = cfg["num_hidden_layers"]
    for i in range(num_layers):
        prefix = f"blk.{i}"
        is_ssm = find_tensor(r, f"{prefix}.ssm_a") is not None
        w = {}
        w["attn_norm"] = get_vector(r, f"{prefix}.attn_norm.weight")
        w["post_attn_norm"] = get_vector(r, f"{prefix}.post_attention_norm.weight")
        w["ffn_gate"] = get_weight(r, f"{prefix}.ffn_gate.weight")
        w["ffn_up"] = get_weight(r, f"{prefix}.ffn_up.weight")
        w["ffn_down"] = get_weight(r, f"{prefix}.ffn_down.weight")

        if is_ssm:
            w["attn_qkv"] = get_weight(r, f"{prefix}.attn_qkv.weight")
            w["attn_gate"] = get_weight(r, f"{prefix}.attn_gate.weight")
            w["ssm_out"] = get_weight(r, f"{prefix}.ssm_out.weight")
            w["conv1d"] = get_weight(r, f"{prefix}.ssm_conv1d.weight")
            w["alpha"] = get_weight(r, f"{prefix}.ssm_alpha.weight")
            w["beta"] = get_weight(r, f"{prefix}.ssm_beta.weight")
            w["ssm_a"] = get_vector(r, f"{prefix}.ssm_a")
            w["dt_bias"] = get_vector(r, f"{prefix}.ssm_dt.bias")
            w["ssm_norm"] = get_vector(r, f"{prefix}.ssm_norm.weight")
        else:
            w["attn_q"] = get_weight(r, f"{prefix}.attn_q.weight")
            w["attn_k"] = get_weight(r, f"{prefix}.attn_k.weight")
            w["attn_v"] = get_weight(r, f"{prefix}.attn_v.weight")
            w["attn_output"] = get_weight(r, f"{prefix}.attn_output.weight")
            w["attn_q_norm"] = get_vector(r, f"{prefix}.attn_q_norm.weight")
            w["attn_k_norm"] = get_vector(r, f"{prefix}.attn_k_norm.weight")

        w["is_ssm"] = is_ssm
        blocks.append(w)

    print(f"  Weights loaded in {time.time() - t0:.1f}s")

    # Build RoPE cache
    sin_cache, cos_cache, rope_dim = build_rope_cache(cfg)

    # Initialize states
    ssm_states = [SSMState(cfg) for _ in range(num_layers)]
    kv_caches = [KVCache(cfg) for _ in range(num_layers)]

    outputs = {}

    # Embedding lookup
    x = embed[token_id, :].astype(np.float32)  # (hidden,)
    outputs["embed"] = x.copy()
    print(f"  Embed: norm={np.linalg.norm(x):.6f}")

    pos = 0  # single token, position 0

    # Per-layer forward
    for i in range(num_layers):
        blk = blocks[i]
        eps = cfg["rms_norm_eps"]

        # Branch 1: SSM or Attention
        x_norm1 = rmsnorm(x, blk["attn_norm"], eps)

        if blk["is_ssm"]:
            branch_out = ssm_process(blk, x_norm1, ssm_states[i], cfg, blk)
            kind = "SSM"
        else:
            branch_out = attn_process(
                blk,
                x_norm1,
                kv_caches[i],
                pos,
                cfg,
                blk,
                sin_cache,
                cos_cache,
                rope_dim,
            )
            kind = "ATN"

        x = x + branch_out
        outputs[f"layer{i}_post_branch"] = x.copy()

        # Branch 2: MLP
        x_norm2 = rmsnorm(x, blk["post_attn_norm"], eps)
        mlp_out = mlp_process(x_norm2, cfg, blk)
        x = x + mlp_out
        outputs[f"layer{i}_post_mlp"] = x.copy()

        print(
            f"  Layer {i:2d} ({kind}): branch_norm={np.linalg.norm(branch_out):.6f} "
            f"mlp_norm={np.linalg.norm(mlp_out):.6f} x_norm={np.linalg.norm(x):.6f}"
        )

    # Final norm
    hidden = rmsnorm(x, output_norm, eps)
    outputs["final_norm"] = hidden.copy()

    # Logits
    logits = lm_head @ hidden
    outputs["logits"] = logits.copy()

    top5_idx = np.argsort(logits)[-5:][::-1]
    print(
        f"  Logits: top-5 ids = {top5_idx.tolist()}, top-5 vals = {logits[top5_idx].tolist()}"
    )

    return outputs


# ── Main ────────────────────────────────────────────────────────────────


def main():
    token_id = DEFAULT_TOKEN_ID
    for arg in sys.argv[1:]:
        if arg.startswith("--token-id="):
            token_id = int(arg.split("=")[1])
        elif arg.startswith("--token-id"):
            idx = sys.argv.index(arg)
            token_id = int(sys.argv[idx + 1])

    print(f"Loading GGUF: {GGUF_PATH}")
    r = load_gguf(GGUF_PATH)
    cfg = get_config(r)

    print(
        f"Config: hidden={cfg['hidden_size']}, layers={cfg['num_hidden_layers']}, "
        f"heads={cfg['num_attention_heads']}, head_dim={cfg['head_dim']}"
    )
    print(
        f"  SSM: inner={cfg['ssm_inner_size']}, state={cfg['ssm_state_size']}, "
        f"groups={cfg['ssm_group_count']}, conv_k={cfg['ssm_conv_kernel']}"
    )
    print(f"  Token ID: {token_id}")

    t0 = time.time()
    outputs = forward_pass(r, token_id, cfg)
    print(f"\nForward pass complete in {time.time() - t0:.1f}s")

    # Save outputs as float32 for easier comparison with Julia Float16
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, arr in outputs.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.npy")
        np.save(path, arr.astype(np.float32))
    print(f"\nSaved {len(outputs)} arrays to {OUTPUT_DIR}/")
    for name in sorted(outputs.keys()):
        arr = outputs[name]
        print(f"  {name:30s} shape={str(arr.shape):20s} norm={np.linalg.norm(arr):.8f}")


if __name__ == "__main__":
    main()
