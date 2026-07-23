#!/usr/bin/env python3
"""Verify a converted repo: shard integrity vs source, and that mlx-lm and
mlx-vlm both produce sane, mutually consistent logits."""
import json
import random
import sys

import mlx.core as mx
import numpy as np

SRC = "/Volumes/models/defiant-fable/src"
PROMPT = (
    "<|im_start|>user\nName the three primary colors.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def rev(k):
    if k.startswith("language_model.model."):
        return "model.language_model." + k[len("language_model.model.") :]
    if k.startswith("vision_tower."):
        return "model.visual." + k[len("vision_tower.") :]
    if k == "language_model.lm_head.weight":
        return "lm_head.weight"
    return k


def check_integrity(M, quantized):
    """Unquantized tensors must still match the source bit-for-bit."""
    wo = json.load(open(M + "/model.safetensors.index.json"))["weight_map"]
    ws = json.load(open(SRC + "/model.safetensors.index.json"))["weight_map"]
    random.seed(0)
    pool = [k for k in wo if rev(k) in ws and not k.endswith((".scales", ".biases"))]
    # in a quantized repo the projections are packed uint32; only 1-D tensors
    # and the conv/norms stay comparable
    if quantized:
        pool = [k for k in pool if k.endswith("norm.weight") or "conv1d" in k]
    picks = random.sample(pool, min(10, len(pool)))
    bad = 0
    for k in picks:
        a = mx.load(M + "/" + wo[k])[k]
        b = mx.load(SRC + "/" + ws[rev(k)])[rev(k)]
        if a.shape != b.shape:  # conv1d layout differences shouldn't happen now
            print(f"  SHAPE {k}: {a.shape} vs {b.shape}")
            bad += 1
            continue
        a32 = np.array(a.astype(mx.float32)).ravel()[:8192]
        b32 = np.array(b.astype(mx.float32)).ravel()[:8192]
        if not np.array_equal(a32, b32):
            cos = float(a32 @ b32 / (np.linalg.norm(a32) * np.linalg.norm(b32) + 1e-30))
            print(f"  MISMATCH {k}  cos={cos:.6f}")
            bad += 1
    print(f"  integrity: {len(picks) - bad}/{len(picks)} sampled tensors exact")
    return bad == 0


def check_loaders(M):
    from mlx_lm.utils import load as lm_load
    from mlx_vlm import load as vlm_load

    lm_model, tok = lm_load(M)
    ids = tok.encode(PROMPT)
    x = mx.array([ids])
    l1 = lm_model(x)
    mx.eval(l1)
    del lm_model

    vm, _ = vlm_load(M)
    l2 = vm.language_model(x)
    if hasattr(l2, "logits"):
        l2 = l2.logits
    mx.eval(l2)
    del vm

    a = np.array(l1[0, -1].astype(mx.float32))
    b = np.array(l2[0, -1].astype(mx.float32))
    t1 = [tok.decode([int(t)]) for t in np.array(mx.argsort(-l1[0, -1])[:5])]
    t2 = [tok.decode([int(t)]) for t in np.array(mx.argsort(-l2[0, -1])[:5])]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"  mlx-lm  top5: {t1}")
    print(f"  mlx-vlm top5: {t2}")
    print(f"  cos(logits): {cos:.6f}")
    ok = cos > 0.99 and not np.isnan(cos) and t1[0] == t2[0]
    return ok


if __name__ == "__main__":
    M = sys.argv[1].rstrip("/")
    quantized = "bf16" not in M
    print(f"== {M.split('/')[-1]}")
    a = check_integrity(M, quantized)
    b = check_loaders(M)
    print(f"  RESULT: {'PASS' if (a and b) else 'FAIL'}")
    sys.exit(0 if (a and b) else 1)
