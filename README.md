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

Upstream fixed the unconditional shift in **0.6.6** ([Blaizzy/mlx-vlm#1556](https://github.com/Blaizzy/mlx-vlm/issues/1556)).
`fix_norms.py` is still worth running, because it makes the output correct on *every*
loader version rather than only the newest: it rewrites norms unshifted and
`conv1d.weight` in raw HF layout `(C, 1, K)`. The unsanitized conv1d trips the heuristic
in both libraries, so mlx-vlm 0.6.4, mlx-vlm 0.6.6 and mlx-lm each apply the shift
exactly once. Verified on all three.

**It buys compatibility, not precision.** `bf16(w + 1)` does round — source `0.13769531`
becomes `1.140625` — but both loaders redo that same bf16 addition at load time, so
storing norms unshifted does not preserve the extra bits. Measured directly: nightmedia's
mxfp4 as published and the same repo with exact norms restored produce **bit-identical
in-memory norms** and identical perplexity to 6 significant figures. Do not expect a
quality gain from this script.

While debugging this I also filed [Blaizzy/mlx-vlm#1665](https://github.com/Blaizzy/mlx-vlm/issues/1665) —
`BPEStreamingDetokenizer.add_token()` strict-decodes UTF-8 and crashes on byte-fallback
tokens, which is what turned the numerical bug into an opaque `UnicodeDecodeError`.

## Benchmark: all tiers vs the bf16 reference

Every model here quantizes the identical bf16 weights, so bf16 is exact ground truth and
the quantization scheme is the only variable. 65,536 tokens of wikitext-2 test at 1024
context, all models fed the same token ids through mlx-lm (`bench.py`), M3 Ultra.

| model | lang. weights | ppl | Δppl | KL(bf16‖q) | top-1 vs bf16 | prefill | decode |
|---|---|---|---|---|---|---|---|
| bf16 (reference) | 17.91 GB | 8.1273 | — | — | — | 1268 t/s | 38.9 t/s |
| **8bit** (affine g64) | 9.51 GB | 8.1277 | +0.00% | 0.00124 | 98.24% | 1354 t/s | 67.3 t/s |
| **6bit** (affine g64) | 7.28 GB | 8.1426 | +0.19% | 0.00523 | 96.20% | 1369 t/s | 80.4 t/s |
| **4bit** (affine g64) | 5.04 GB | 8.5816 | +5.59% | 0.07330 | 87.06% | 1436 t/s | 109.1 t/s |
| nightmedia mxfp4 (g32) | 4.76 GB | 8.9443 | +10.05% | 0.11328 | 82.34% | 1430 t/s | 113.8 t/s |

**8bit is free.** It matches bf16 perplexity to four decimals (+0.00%) at KL 0.00124 and
98.24% argmax agreement, for 47% of the weight footprint and 1.7x the decode speed. If
you have the RAM there is no quality reason to run bf16.

**6bit is the sweet spot.** +0.19% perplexity and 96.2% agreement for 7.3 GB — a fifth of
8bit's already-small error budget spent to save 2.2 GB and gain 20% decode speed.

**The cliff is between 6bit and 4bit**, not where the even spacing suggests. KL jumps 14x
(0.0052 → 0.0733) and perplexity goes from +0.19% to +5.59%. Anyone who can afford 7.3 GB
should not be running 4bit.

**4bit vs MXFP4**: at the 4-bit tier our affine g64 loses roughly *half* the perplexity
MXFP4 does (+5.59% vs +10.05%), holds 35% lower KL, and matches bf16's argmax 4.7 points
more often. That is not free — affine g64 spends 4.5 bits/weight against MXFP4's 4.25 (a
shared E8M0 scale per 32 values), so ~6% more storage and ~4% slower decode buys it.

Both keep the vision tower at bf16 (0.91 GB each); it is unused in a text-only perplexity
run, so it does not affect the quality columns. Prefill is memory-bandwidth-bound and
nearly flat across tiers; decode scales with weight size as expected.

Reproduce with `python3 bench.py --tokens 65536 --seq-len 1024`. Perplexity is
corpus-dependent, so compare the columns against each other rather than against other
write-ups.

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
| `bench.py` | ppl / KL / top-1 / throughput vs the bf16 reference |
| `publish.py` | Renders the model card per quant and uploads to the Hub |
| `card_template.md` | Model card template |

## Notes

- MTP weights are dropped — no MLX loader uses them. That costs only the speculative
  decoding speedup the GGUF "MTP" variants advertise, not quality.
- The vision tower stays bf16 in every tier, which is why 4bit measures 5.06 bits/weight.
- Sampling: temperature ≤ 1.0, repetition penalty 1.0 (off). Raising either hurts.

## License

Tooling: MIT. The models are Apache 2.0, inherited from Qwen3.5-9B.
