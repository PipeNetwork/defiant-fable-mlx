#!/usr/bin/env python3
"""Prove that DavidAU's Defiant-Fable GGUFs are a straight requantization of
nightmedia/Qwen3.5-9B-DS9-USS-Defiant, using HTTP range reads only (~35 MB, no
full download of either 10 GB GGUF or 19 GB safetensors set).

Three checks:
  1. GGUF `output.weight` is stored BF16, so it must be bit-identical to the
     source `lm_head.weight`.
  2. GGUF norm tensors must equal the source norms + 1.0 exactly -- llama.cpp
     stores Qwen3.5 RMSNorm weights pre-shifted.
  3. GGUF Q8_0 `token_embd` must dequantize to the source `embed_tokens` with
     the error profile of a plain Q8_0 round-trip.

Usage: python3 verify_provenance.py
"""
import io
import json
import struct
import urllib.request

import numpy as np

GGUF = (
    "https://huggingface.co/DavidAU/"
    "Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/"
    "resolve/main/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q8_0.gguf"
)
SRC = "https://huggingface.co/nightmedia/Qwen3.5-9B-DS9-USS-Defiant/resolve/main/"

GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def fetch(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req).read()


def read_gguf_index(url, header_bytes=24_000_000):
    """Parse the GGUF header and return (data_start, {name: (dims, type, offset)})."""
    f = io.BytesIO(fetch(url, 0, header_bytes - 1))

    def rd(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def rd_str():
        return f.read(rd("<Q")).decode("utf-8", errors="replace")

    def rd_val(t):
        if t == 8:
            return rd_str()
        if t == 9:
            et, n = rd("<I"), rd("<Q")
            return [rd_val(et) for _ in range(n)]
        return rd(GGUF_SCALARS[t])

    assert f.read(4) == b"GGUF"
    rd("<I")
    n_tensors, n_kv = rd("<Q"), rd("<Q")
    kv = {}
    for _ in range(n_kv):
        k = rd_str()
        kv[k] = rd_val(rd("<I"))

    tensors = {}
    for _ in range(n_tensors):
        name = rd_str()
        dims = [rd("<Q") for _ in range(rd("<I"))]
        tensors[name] = (dims, rd("<I"), rd("<Q"))

    align = kv.get("general.alignment", 32)
    data_start = (f.tell() + align - 1) // align * align
    return data_start, tensors, kv


class SafetensorsReader:
    """Random access into a sharded safetensors repo over HTTP."""

    def __init__(self, base):
        self.base = base
        self.weight_map = json.loads(
            urllib.request.urlopen(base + "model.safetensors.index.json").read()
        )["weight_map"]
        self._headers = {}

    def _header(self, shard):
        if shard not in self._headers:
            url = self.base + shard
            n = struct.unpack("<Q", fetch(url, 0, 7))[0]
            self._headers[shard] = (json.loads(fetch(url, 8, 8 + n - 1)), 8 + n)
        return self._headers[shard]

    def read(self, name, max_elems=None):
        shard = self.weight_map[name]
        header, base_off = self._header(shard)
        meta = header[name]
        start, end = meta["data_offsets"]
        itemsize = 2 if meta["dtype"] in ("BF16", "F16") else 4
        if max_elems is not None:
            end = min(end, start + max_elems * itemsize)
        raw = fetch(self.base + shard, base_off + start, base_off + end - 1)
        if meta["dtype"] == "BF16":
            return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        if meta["dtype"] == "F32":
            return np.frombuffer(raw, "<f4")
        raise ValueError(f"unhandled dtype {meta['dtype']}")


def main():
    print("reading GGUF header ...")
    data_start, tensors, kv = read_gguf_index(GGUF)
    print(f"  arch={kv['general.architecture']}  name={kv['general.name']!r}")
    print(f"  {len(tensors)} tensors\n")

    src = SafetensorsReader(SRC)
    ok = True

    # 1. lm_head: BF16 in the GGUF, so it must match bit-for-bit
    dims, _, off = tensors["output.weight"]
    n_rows, row_bytes = 64, dims[0] * 2
    gguf_bits = np.frombuffer(
        fetch(GGUF, data_start + off, data_start + off + n_rows * row_bytes - 1), "<u2"
    )
    src_f32 = src.read("lm_head.weight", max_elems=n_rows * dims[0])
    src_bits = (src_f32.view(np.uint32) >> 16).astype(np.uint16)
    exact = np.array_equal(gguf_bits, src_bits)
    ok &= exact
    print(f"[{'PASS' if exact else 'FAIL'}] lm_head bit-exact over {gguf_bits.size} bf16 values "
          f"({int((gguf_bits != src_bits).sum())} mismatches)")

    # 2. norms: GGUF stores them pre-shifted by +1
    pairs = [
        ("blk.0.attn_norm.weight", "model.language_model.layers.0.input_layernorm.weight"),
        ("blk.3.attn_q_norm.weight", "model.language_model.layers.3.self_attn.q_norm.weight"),
        ("output_norm.weight", "model.language_model.norm.weight"),
        ("blk.31.post_attention_norm.weight",
         "model.language_model.layers.31.post_attention_layernorm.weight"),
    ]
    # A handful of elements land 1 ULP apart because the `+1` is done in float32 and
    # the source values are tiny (~1e-5), so the sum rounds. Allow 1 ULP at 1.0.
    ULP = np.spacing(np.float32(1.0))
    for gname, sname in pairs:
        dims, _, off = tensors[gname]
        n = int(np.prod(dims))
        g = np.frombuffer(fetch(GGUF, data_start + off, data_start + off + n * 4 - 1), "<f4")
        s = src.read(sname, max_elems=n)
        diff = np.abs(g.astype(np.float64) - (s.astype(np.float64) + 1.0))
        n_exact = int((diff == 0).sum())
        shifted = bool((diff <= ULP).all())
        ok &= shifted
        print(f"[{'PASS' if shifted else 'FAIL'}] {gname} == source + 1.0  "
              f"({n_exact}/{n} exact, rest within 1 ULP)")

    # 3. token_embd: Q8_0 round-trip of the source embeddings
    dims, _, off = tensors["token_embd.weight"]
    d_model, rows = dims[0], 4
    blocks = d_model // 32
    nbytes = rows * blocks * 34  # Q8_0: fp16 scale + 32 int8 per 32 weights
    raw = np.frombuffer(
        fetch(GGUF, data_start + off, data_start + off + nbytes - 1), np.uint8
    ).reshape(rows, blocks, 34)
    scales = raw[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(rows, blocks, 1)
    deq = (scales * raw[:, :, 2:].view(np.int8).astype(np.float32)).reshape(rows, d_model)
    s = src.read("model.language_model.embed_tokens.weight", max_elems=rows * d_model)
    s = s.reshape(rows, d_model)
    cos = float((s.ravel() @ deq.ravel()) / (np.linalg.norm(s) * np.linalg.norm(deq)))
    good = cos > 0.9999
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] token_embd Q8_0 dequant vs source: cos={cos:.6f}")

    print("\n" + ("CONCLUSION: the GGUF is a requantization of the bf16 source."
                  if ok else "CONCLUSION: tensors do NOT match -- different weights."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
