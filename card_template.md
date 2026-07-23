---
language:
- en
- zh
license: apache-2.0
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- apple-silicon
- qwen3.5
- fine tune
- heretic
- uncensored
- abliterated
- merge
- thinking
- reasoning
- creative
- writing
- fiction
- roleplaying
- vision
base_model: nightmedia/Qwen3.5-9B-DS9-USS-Defiant
---

# Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-__QUANT__

MLX conversion of **Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic** for Apple silicon — __QUANT_DESC__

Uncensored ("Heretic'd") multi-stage merge of Qwen3.5-9B fine tunes by [nightmedia](https://huggingface.co/nightmedia) and [DavidAU](https://huggingface.co/DavidAU), with a compacted-but-stronger thinking block. Vision is included and works out of the box — no separate `mmproj` download.

## Provenance

This was converted from **[nightmedia/Qwen3.5-9B-DS9-USS-Defiant](https://huggingface.co/nightmedia/Qwen3.5-9B-DS9-USS-Defiant)** (bfloat16 safetensors), which is the exact same weight set that [DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF](https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF) packages as GGUF.

We verified the identity numerically rather than assuming it:

| Check | Result |
|---|---|
| `lm_head.weight` (BF16 in both) vs GGUF `output.weight` | **bit-exact**, 0 mismatches across 262,144 values |
| RMSNorm weights (`input_layernorm`, `q_norm`, `post_attention_layernorm`, final `norm`) | equal `source + 1.0` under llama.cpp's shift convention (12542/12544 elements exact; 2 differ by 1 ULP from the float32 addition) |
| `embed_tokens` vs GGUF Q8_0 `token_embd` | cosine 0.999956 — consistent with a plain Q8_0 round-trip |

So these MLX quants are made from the original bf16 weights, **not** by dequantizing a GGUF. There is no GGUF round-trip loss.

Credit for the model itself goes to nightmedia and DavidAU; this repo only does the MLX conversion.

## Architecture

Qwen3.5-9B is a hybrid multimodal model:

- **32 layers**, 3:1 ratio of gated-delta linear attention to full attention (`full_attention_interval: 4`)
- Gated output attention (`attn_output_gate`), head_dim 256, 16 Q heads / 4 KV heads
- Interleaved mRoPE with `partial_rotary_factor` 0.25, `rope_theta` 1e7
- **262,144 native context**
- 27-layer vision tower (patch 16, spatial merge 2), 248,320 vocab

The source also ships MTP (multi-token-prediction) weights. MLX does not use them, so they are dropped during conversion — this costs no quality, only the speculative-decoding speedup that the GGUF "MTP" variants offer.

## Usage

Vision + text with `mlx-vlm`:

```bash
pip install mlx-vlm
python -m mlx_vlm.generate \
  --model pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-__QUANT__ \
  --image your_image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 512
```

Text-only with `mlx-lm` (loads the same repo, ignores the vision tower):

```bash
pip install mlx-lm
mlx_lm.generate \
  --model pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-__QUANT__ \
  --prompt "Write the opening paragraph of a noir story set on a space station." \
  --max-tokens 512
```

Python:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-__QUANT__")
config = model.config

prompt = apply_chat_template(processor, config, "Describe this image.", num_images=1)
print(generate(model, processor, prompt, ["your_image.jpg"], max_tokens=512, verbose=False))
```

## All quantizations, measured

Every tier below quantizes the *identical* bf16 weights, so bf16 is exact ground truth and
the quantization scheme is the only variable. Measured on 65,536 tokens of wikitext-2 test
at 1024 context, all models fed the same token ids through mlx-lm, on an M3 Ultra. KL is
against bf16's own output distribution — lower means closer to the original model.

| Repo | Size | Bits/w | ppl | Δppl | KL(bf16‖q) | top-1 | decode | verdict |
|---|---|---|---|---|---|---|---|---|
| [bf16](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-bf16) | __SZ16__ | 16 | 8.1273 | — | — | — | 38.6 t/s | exact reference |
| [8bit](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-8bit) | __SZ8__ | 8.86 | 8.1277 | +0.00% | 0.00124 | 98.24% | 66.9 t/s | free — no reason to run bf16 |
| [6bit](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-6bit) | __SZ6__ | 6.96 | 8.1426 | +0.19% | 0.00523 | 96.20% | 80.3 t/s | near-lossless |
| [5bit](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-5bit) | __SZ5__ | 6.01 | 8.2012 | +0.91% | 0.01845 | 93.24% | 91.5 t/s | **best quality-per-GB** |
| [4bit](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-4bit) | __SZ4__ | 5.06 | 8.5816 | +5.59% | 0.07330 | 87.06% | 108.6 t/s | usable floor; common default |
| [3bit](https://huggingface.co/pipenetwork/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX-3bit) | __SZ3__ | 4.11 | 10.8167 | +33.09% | 0.32843 | 73.44% | 124.9 t/s | tight-memory fallback only |
| [nightmedia mxfp4](https://huggingface.co/nightmedia/Qwen3.5-9B-DS9-USS-Defiant-mxfp4-mlx) | 5.6 GB | 4.25 | 8.9443 | +10.05% | 0.11328 | 82.34% | 113.5 t/s | for comparison |

Sizes are the full repo; `Bits/w` is the whole-model average, which sits above the nominal
width because the vision tower stays bf16 (0.91 GB) in every tier. All tiers are group size
64, `affine` mode.

Reading it:

- **8bit is effectively free** — bf16 perplexity to four decimals at 47% of the footprint.
- **5bit is the best quality-per-GB**: under 1% perplexity.
- **The usable floor is 4bit.** 3bit still answers factual questions correctly but costs
  +33% perplexity; treat it as a tight-memory fallback.
- **No 2-bit tier is published.** Pure 2-bit collapses (ppl 214.5, 28% top-1) and MLX's
  `mixed_2_6` recipe, while grammatical, still runs ~7x bf16 perplexity at 3.94 GB — no
  smaller than 3bit and 5x worse. Both were built and measured; neither is usable.
- At the 4-bit tier this affine group-64 quant loses about half the perplexity MXFP4 does,
  costing 4.5 vs 4.25 bits/weight.

Reproduce with [`bench.py`](https://github.com/PipeNetwork/defiant-fable-mlx).

## Sampling

DavidAU's notes for this model, which carry over:

- Temperature 1.0 or below works best; higher temps degrade coherence
- Repetition penalty 1.0 (off) — raising it hurts this model
- The thinking block is compacted; give it room with a generous `max-tokens`

## Benchmarks

From the source model card (source weights, non-MLX quant tiers), for reference:

```
          arc/c  arc/e boolq hswag obkqa piqa  wino
bf16      0.649, 0.832, 0.895, 0.713, 0.482, 0.783, 0.699
mxfp8     0.647, 0.836, 0.895, 0.706, 0.460, 0.784, 0.695
mxfp4     0.640, 0.824, 0.886, 0.703, 0.468, 0.780, 0.691

Qwen3.5-9B-Instruct (base, non-heretic)
mxfp8     0.571, 0.719, 0.895, 0.683, 0.426, 0.770, 0.671
```

These are not measurements of these MLX repos — treat them as characterising the weights, not this quantization.

## License

Apache 2.0, inherited from Qwen3.5-9B.

This model has had its safety post-training removed and will follow instructions without refusal. You are responsible for how you use it.
