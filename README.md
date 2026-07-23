# defiant-fable-mlx

MLX conversion pipeline for **Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic** — a
vision-enabled, uncensored Qwen3.5-9B merge — targeting Apple silicon.

Published models:

| Repo | Size | Bits/weight |
|---|---|---|
| [`pipenetwork/…-MLX-4bit`](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-4bit) | 6.0 GB | 5.06 |
| [`pipenetwork/…-MLX-6bit`](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-6bit) | 7.7 GB | 6.96 |
| [`pipenetwork/…-MLX-8bit`](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-8bit) | 9.7 GB | 8.86 |
| [`pipenetwork/…-MLX-bf16`](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-bf16) | 18.8 GB | 16 |

All four keep the 27-layer vision tower at bf16, so they load in **mlx-vlm** (image+text)
and **mlx-lm** (text-only) alike. On an M3 Ultra the 4bit runs ~110 tok/s at 5.2 GB peak.

## Provenance: the GGUF is not a distinct model

[`DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF`](https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF)
ships GGUF only (max Q8_0) with no safetensors, which would normally force a lossy
GGUF→safetensors dequantization. It turns out not to be necessary: those GGUFs are a
straight requantization of [`nightmedia/Qwen3.5-9B-DS9-USS-Defiant`](https://huggingface.co/nightmedia/Qwen3.5-9B-DS9-USS-Defiant) (bf16).

`verify_provenance.py` proves it by range-reading both repos over HTTP — no full download:

| Check | Result |
|---|---|
| GGUF `output.weight` (BF16) vs source `lm_head.weight` | **bit-exact**, 0/262144 mismatches |
| RMSNorms vs source | match `source + 1.0` under llama.cpp's shift convention (12542/12544 exact; 2 elements 1 ULP off from the float32 addition) |
| GGUF Q8_0 `token_embd` vs source `embed_tokens` | cosine 0.999956 (a plain Q8_0 round-trip) |

So these quants come from the original bf16 weights and carry no GGUF round-trip loss.
The "Defiant-Fable" name is DavidAU's rebrand plus NEO-imatrix quants; the GGUF metadata
gives it away with `general.name = 'Qwen3.5 9B DS9 USS Defiant NM'`.

Credit for the model belongs to [nightmedia](https://huggingface.co/nightmedia) and
[DavidAU](https://huggingface.co/DavidAU) — this repo only does the MLX conversion.

## The norm-shift trap

Qwen3.5 stores RMSNorm weights such that the loader must apply `w → w + 1`.
mlx-lm guards that shift behind `has_mtp_weights or has_unsanitized_conv1d`;
**mlx-vlm ≤ 0.6.5 applied it unconditionally**, while its own converter had already
baked the shift into the saved file. The result was a double shift:

- cos(logits) ≈ 0.13 against mlx-lm, first sampled token an invalid UTF-8 byte
- mlx-lm loaded the *same* repo correctly, so the breakage was easy to miss
- as a bonus, `bf16(w + 1)` is lossy — source `0.13769531` becomes `1.140625`, so you
  cannot recover the original by subtracting 1

Upstream fixed the unconditional shift in **0.6.6** ([Blaizzy/mlx-vlm#1556](https://github.com/Blaizzy/mlx-vlm/issues/1556)).
`fix_norms.py` is still worth running, because it makes the output correct on *every*
loader version rather than only the newest: it rewrites norms unshifted and
`conv1d.weight` in raw HF layout `(C, 1, K)`, taking exact values from the source
checkpoint. The unsanitized conv1d trips the heuristic in both libraries, so mlx-vlm
0.6.4, mlx-vlm 0.6.6 and mlx-lm each apply the shift exactly once. Verified on all three.

While debugging this I also filed [Blaizzy/mlx-vlm#1665](https://github.com/Blaizzy/mlx-vlm/issues/1665) —
`BPEStreamingDetokenizer.add_token()` strict-decodes UTF-8 and crashes on byte-fallback
tokens, which is what turned the numerical bug into an opaque `UnicodeDecodeError`.

## Usage

```bash
hf download nightmedia/Qwen3.5-9B-DS9-USS-Defiant --local-dir /path/to/src
rm -rf /path/to/src/.cache          # mlx-vlm copytree's every subdir into each output

# nightmedia's repo omits these; take them from the upstream base model
for f in preprocessor_config.json video_preprocessor_config.json; do
  curl -sfL "https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/$f" -o "/path/to/src/$f"
done

./build.sh                          # convert -> fix_norms -> verify, per quant
python3 publish.py 4bit 6bit 8bit bf16
```

`build.sh` refuses to move on until `verify.py` passes, which requires unquantized
tensors to still match the source bit-for-bit **and** mlx-lm and mlx-vlm to agree on the
logits (cos ≈ 1.0). Measured: 1.000000 / 0.999950 / 1.000000 / 1.000000.

Paths in the scripts point at `/Volumes/models/defiant-fable`; edit `ROOT` to relocate.

## Files

| File | Purpose |
|---|---|
| `verify_provenance.py` | Proves the GGUF is a requant of the bf16 source, via HTTP range reads |
| `build.sh` | convert → fix_norms → verify for bf16/8bit/6bit/4bit |
| `fix_norms.py` | Rewrites norms + conv1d in raw HF convention so every loader shifts once |
| `verify.py` | Gates on source integrity + mlx-lm/mlx-vlm logit agreement |
| `publish.py` | Renders the model card per quant and uploads to the Hub |
| `card_template.md` | Model card template |

## Notes

- MTP weights are dropped — no MLX loader uses them. That costs only the speculative
  decoding speedup the GGUF "MTP" variants advertise, not quality.
- The vision tower stays bf16 in every tier, which is why 4bit measures 5.06 bits/weight.
- Sampling: temperature ≤ 1.0, repetition penalty 1.0 (off). Raising either hurts.

## License

Tooling: MIT. The models are Apache 2.0, inherited from Qwen3.5-9B.
