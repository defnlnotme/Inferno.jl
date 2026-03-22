#!/usr/bin/env python3
"""
Test runner: generates reference values for ALL components.
Run: python3 tests/python/generate_all.py
Then: julia --project=. tests/julia/verify_all.jl
"""

import sys

sys.path.insert(0, "tests/python")
from common import *
import os

r = load_gguf()
cfg = get_config(r)
eps = cfg["rms_norm_eps"]
OUT = "/tmp/test_refs"
os.makedirs(OUT, exist_ok=True)

# Load embedding
emb = load_weight(r, "token_embd.weight")
x = emb[TOKEN_ID, :].astype(np.float64)
np.save(f"{OUT}/01_embed.npy", x)
print(f"  01_embed norm={np.linalg.norm(x):.6f}")

# Test RMSNorm
norm_w = load_weight(r, "blk.0.attn_norm.weight").astype(np.float64)
ss = np.sum(x**2)
scale = 1.0 / np.sqrt(ss / len(x) + eps)
x_norm = (x * scale * norm_w).astype(np.float32)
np.save(f"{OUT}/02_rmsnorm.npy", x_norm)
print(f"  02_rmsnorm norm={np.linalg.norm(x_norm):.6f}")

# Test SSM block (layer 0) — QKV + conv + recurrence + output
blk = "blk.0"
qkv = (load_weight(r, f"{blk}.attn_qkv.weight").astype(np.float64)) @ x
gate = (load_weight(r, f"{blk}.attn_gate.weight").astype(np.float64)) @ x
K = cfg["ssm_conv_kernel"]
conv_state = np.zeros((qkv.shape[0], K), np.float64)
conv_state[:, -1] = qkv
conv1d_w = load_weight(r, f"{blk}.ssm_conv1d.weight").astype(np.float64)
x_conv = sum(conv_state[:, k] * conv1d_w[:, k] for k in range(K))
x_conv = silu(x_conv)
np.save(f"{OUT}/03_ssm_conv.npy", x_conv)

# SSM recurrence
ss, grp = cfg["ssm_state_size"], cfg["ssm_group_count"]
hvd = cfg["ssm_inner_size"] // cfg["ssm_time_step_rank"]
qk_size = ss * grp
q_all = x_conv[:qk_size].reshape(grp, ss)
k_all = x_conv[qk_size : 2 * qk_size].reshape(grp, ss)
v_all = x_conv[2 * qk_size : 2 * qk_size + cfg["ssm_inner_size"]].reshape(grp, hvd)
alpha = load_weight(r, f"{blk}.ssm_alpha.weight").astype(np.float64) @ x
beta = load_weight(r, f"{blk}.ssm_beta.weight").astype(np.float64) @ x
dt_b = load_weight(r, f"{blk}.ssm_dt.bias").astype(np.float64)
ssm_a = load_weight(r, f"{blk}.ssm_a").astype(np.float64)
h = np.zeros((hvd, ss), np.float64)
y_all = np.zeros(cfg["ssm_inner_size"], np.float64)
for g in range(grp):
    qg, kg, vg = q_all[g], k_all[g], v_all[g]
    qn = qg / np.sqrt(np.sum(qg**2) + eps)
    kn = kg / np.sqrt(np.sum(kg**2) + eps)
    dg = np.exp(np.log1p(np.exp(alpha[g] + dt_b[g])) * ssm_a[g])
    bg = 1.0 / (1.0 + np.exp(-beta[g]))
    h *= dg
    h += np.outer(bg * (vg - h @ kn), kn)
    y_all[g * hvd : (g + 1) * hvd] = h @ qn
np.save(f"{OUT}/04_ssm_recurrence.npy", y_all)

# SSM output norm + gate + projection
sn = load_weight(r, f"{blk}.ssm_norm.weight").astype(np.float64)
y_r = y_all.reshape(grp, hvd)
for g in range(grp):
    sc = 1.0 / np.sqrt(np.sum(y_r[g] ** 2) / hvd + eps)
    y_r[g] *= sc * sn
z_g = silu(gate)
y_r *= z_g.reshape(grp, hvd)
branch = load_weight(r, f"{blk}.ssm_out.weight").astype(np.float64) @ y_r.flatten()
np.save(f"{OUT}/05_ssm_branch.npy", branch)

# Layer 0 full (branch + mlp)
x_post_branch = x + branch
np.save(f"{OUT}/06_layer0_branch.npy", x_post_branch)
x_norm2 = rmsnorm(
    x_post_branch.astype(np.float32),
    load_weight(r, f"{blk}.post_attention_norm.weight"),
    eps,
)
g_m = silu(load_weight(r, f"{blk}.ffn_gate.weight") @ x_norm2)
u_m = load_weight(r, f"{blk}.ffn_up.weight") @ x_norm2
mlp = load_weight(r, f"{blk}.ffn_down.weight").astype(np.float64) @ (g_m * u_m).astype(
    np.float64
)
x_layer0 = x_post_branch + mlp
np.save(f"{OUT}/07_layer0_mlp.npy", x_layer0)

# Attention block (layer 3)
blk3 = "blk.3"
x3 = x_layer0
x3_norm = rmsnorm(
    x3.astype(np.float32), load_weight(r, f"{blk3}.attn_norm.weight"), eps
)
hd = cfg["head_dim"]
nq = cfg["num_attention_heads"]
nkv = cfg["num_key_value_heads"]
q_buf = load_weight(r, f"{blk3}.attn_q.weight") @ x3_norm
k_buf = load_weight(r, f"{blk3}.attn_k.weight") @ x3_norm
v_buf = load_weight(r, f"{blk3}.attn_v.weight") @ x3_norm
np.save(f"{OUT}/08_attn_qkv.npy", np.concatenate([q_buf, k_buf, v_buf]))

# Full forward pass
sys.path.insert(0, "examples/checks")
from forward_ref import forward_pass

outputs = forward_pass(r, TOKEN_ID, cfg)
np.save(f"{OUT}/09_final_norm.npy", outputs["final_norm"])
np.save(f"{OUT}/10_logits.npy", outputs["logits"])

top5 = np.argsort(outputs["logits"])[-5:][::-1]
print(f"  09_final_norm norm={np.linalg.norm(outputs['final_norm']):.6f}")
print(f"  10_logits top-5={top5.tolist()}")

# Verify dequantization
from gen_dequant_ref import *

print(f"\n  All references saved to {OUT}/")
