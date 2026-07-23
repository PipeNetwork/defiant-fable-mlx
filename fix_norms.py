#!/usr/bin/env python3
"""Make an mlx-vlm-converted Qwen3.5 repo loadable by BOTH mlx-vlm and mlx-lm.

mlx-vlm's Qwen3_5 `sanitize()` adds +1.0 to RMSNorm weights unconditionally, but
its own converter has already baked that shift into the saved file -- so loading
an mlx-vlm-converted repo with mlx-vlm double-shifts and produces garbage.
mlx-lm guards the same shift behind `has_mtp_weights or has_unsanitized_conv1d`.

Fix: write the affected tensors back in the *raw HF* convention taken straight
from the source checkpoint --
  * norm weights unshifted (also avoids the bf16 rounding loss of `w + 1`)
  * conv1d.weight in HF layout (C, 1, K) instead of the moveaxis'd (C, K, 1)
The unsanitized conv1d makes mlx-lm's heuristic fire, so both loaders now apply
the +1 exactly once.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def src_key(out_key: str) -> str:
    """Map an mlx-vlm sanitized key back to its source-checkpoint name."""
    if out_key.startswith("language_model.model."):
        return "model.language_model." + out_key[len("language_model.model.") :]
    if out_key.startswith("vision_tower."):
        return "model.visual." + out_key[len("vision_tower.") :]
    if out_key == "language_model.lm_head.weight":
        return "lm_head.weight"
    return out_key


def load_source(src: Path):
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {}
    for name, shard in wm.items():
        shards.setdefault(shard, []).append(name)
    return wm, shards


def main(out_dir: str, src_dir: str):
    out, src = Path(out_dir), Path(src_dir)
    wm_src, _ = load_source(src)

    index = json.loads((out / "model.safetensors.index.json").read_text())
    wm_out = index["weight_map"]

    targets = [
        k
        for k in wm_out
        if any(k.endswith(s) for s in NORM_SUFFIXES) or "conv1d.weight" in k
    ]
    print(f"{out.name}: restoring {len(targets)} tensors to raw HF convention")

    # group the tensors we need by their source shard so each shard loads once
    need = {}
    for k in targets:
        need.setdefault(wm_src[src_key(k)], []).append(k)

    replacement = {}
    for shard, keys in need.items():
        blob = mx.load(str(src / shard))
        for k in keys:
            v = blob[src_key(k)]
            if "conv1d.weight" in k:
                # keep HF's (C, 1, K); both loaders moveaxis it themselves
                assert v.shape[-1] != 1, f"{k} already sanitized in source"
            replacement[k] = v
        del blob

    # rewrite only the shards that actually contain a replaced tensor
    by_shard = {}
    for k in targets:
        by_shard.setdefault(wm_out[k], []).append(k)

    for shard, keys in by_shard.items():
        path = out / shard
        blob = dict(mx.load(str(path)))
        for k in keys:
            blob[k] = replacement[k].astype(blob[k].dtype)
        # mx.load is lazy and memory-maps `path`; writing back to it in place
        # corrupts every tensor still backed by that mapping. Materialize, then
        # write to a temp file and swap it in.
        mx.eval(list(blob.values()))
        # mlx appends `.safetensors` to any path lacking it, so keep it last
        tmp = path.with_name(path.stem + ".tmp.safetensors")
        mx.save_safetensors(str(tmp), blob, metadata={"format": "mlx"})
        del blob
        tmp.replace(path)
        print(f"  rewrote {shard} ({len(keys)} tensors)")

    print("  done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
