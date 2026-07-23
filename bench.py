#!/usr/bin/env python3
"""Compare quantizations of the same weights against their bf16 reference.

Both quants derive from nightmedia/Qwen3.5-9B-DS9-USS-Defiant, so bf16 is exact
ground truth and the quantization scheme is the only variable. Reports:

  * perplexity on a fixed wikitext-2 slice (lower is better)
  * KL(bf16 || quant) per token -- how far the quantized distribution drifts
    from the original (lower is better); the sharpest quant-fidelity metric
  * top-1 agreement with bf16's argmax
  * decode throughput and peak memory, measured separately

All three models are held in memory and fed identical token ids in lockstep, so
no reference logprobs need to be buffered (that would be ~32 GB) and every model
sees exactly the same inputs through the same mlx-lm code path.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path("/Volumes/models/defiant-fable")
NAME = "Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX"

REF = "bf16 (reference)"
QUANTS = ["pipenetwork 4bit (affine g64)", "nightmedia mxfp4 (g32)"]
PATHS = {
    REF: ROOT / f"{NAME}-bf16",
    "pipenetwork 4bit (affine g64)": ROOT / f"{NAME}-4bit",
    "nightmedia mxfp4 (g32)": ROOT / "mxfp4",
}


def build_corpus(tokenizer, n_tokens, seq_len):
    text = (ROOT / "wikitext2_test.txt").read_text()
    ids = tokenizer.encode(text)[: n_tokens + 1]
    n_seq = (len(ids) - 1) // seq_len
    if n_seq == 0:
        raise SystemExit("corpus too short")
    x = np.array(ids[: n_seq * seq_len], np.int32).reshape(n_seq, seq_len)
    y = np.array(ids[1 : n_seq * seq_len + 1], np.int32).reshape(n_seq, seq_len)
    return mx.array(x), mx.array(y), n_seq


def logprobs_of(model, xi):
    logits = model(xi).astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def measure_speed(path, prompt_tokens=512, gen_tokens=128, repeats=3):
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load

    mx.reset_peak_memory()
    model, tok = load(str(path))
    ids = (tok.encode("The history of spaceflight is a story of ") * 300)[:prompt_tokens]
    prompt = mx.array(ids, dtype=mx.int32)

    best_prefill, best_decode = 0.0, 0.0
    for _ in range(repeats):
        mx.clear_cache()
        t0 = time.perf_counter()
        it = generate_step(prompt, model, max_tokens=gen_tokens)
        tok0 = next(it)
        mx.eval(tok0[0])
        prefill_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        n = 0
        for out in it:
            mx.eval(out[0])
            n += 1
        decode_s = time.perf_counter() - t1
        best_prefill = max(best_prefill, prompt_tokens / prefill_s)
        if n and decode_s > 0:
            best_decode = max(best_decode, n / decode_s)

    peak = mx.get_peak_memory() / 1e9
    del model
    gc.collect()
    mx.clear_cache()
    return {"prefill_tps": best_prefill, "decode_tps": best_decode, "speed_peak_gb": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()

    from mlx_lm.utils import load

    for label, p in PATHS.items():
        if not p.exists():
            raise SystemExit(f"missing {label}: {p}")

    print("loading models ...")
    models, tokenizers = {}, {}
    for label, p in PATHS.items():
        models[label], tokenizers[label] = load(str(p))

    # identical vocab is what makes feeding shared token ids valid
    vocabs = {label: len(t.vocab) for label, t in tokenizers.items()}
    assert len(set(vocabs.values())) == 1, f"vocab mismatch: {vocabs}"
    print(f"  vocab size {next(iter(vocabs.values()))} identical across all models")

    x, y, n_seq = build_corpus(tokenizers[REF], args.tokens, args.seq_len)
    print(f"  corpus: {n_seq} x {args.seq_len} = {n_seq * args.seq_len} tokens\n")

    stats = {
        label: {"nll": 0.0, "kl": 0.0, "agree": 0, "n": 0}
        for label in PATHS
    }

    for i in range(n_seq):
        xi, yi = x[i : i + 1], y[i : i + 1]
        ref_lp = logprobs_of(models[REF], xi)
        mx.eval(ref_lp)
        ref_top = ref_lp.argmax(axis=-1)
        ref_p = mx.exp(ref_lp)

        for label in PATHS:
            lp = ref_lp if label == REF else logprobs_of(models[label], xi)
            tgt = mx.take_along_axis(lp, yi[..., None], axis=-1).squeeze(-1)
            mx.eval(tgt)
            s = stats[label]
            s["nll"] += float(-tgt.sum().item())
            s["n"] += tgt.size
            if label != REF:
                kl = (ref_p * (ref_lp - lp)).sum(axis=-1)
                agree = (lp.argmax(axis=-1) == ref_top).sum()
                mx.eval(kl, agree)
                s["kl"] += float(kl.sum().item())
                s["agree"] += int(agree.item())
                del lp
            del tgt
        del ref_lp, ref_p, ref_top
        mx.clear_cache()
        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/{n_seq} sequences")

    results = {}
    for label, s in stats.items():
        r = {"ppl": float(np.exp(s["nll"] / s["n"])), "tokens": s["n"]}
        if label != REF:
            r["kl"] = s["kl"] / s["n"]
            r["top1_agree"] = s["agree"] / s["n"]
        results[label] = r

    del models
    gc.collect()
    mx.clear_cache()

    if not args.skip_speed:
        print("\nmeasuring speed (models loaded one at a time) ...")
        for label, p in PATHS.items():
            results[label].update(measure_speed(p))
            print(f"  {label}: {results[label]['decode_tps']:.1f} tok/s decode")

    for label, p in PATHS.items():
        results[label]["disk_gb"] = sum(
            f.stat().st_size for f in p.glob("*.safetensors")
        ) / 1e9

    (ROOT / "bench_results.json").write_text(json.dumps(results, indent=2))

    ref_ppl = results[REF]["ppl"]
    print("\n" + "=" * 104)
    print(f"{'model':<31}{'size':>7}{'ppl':>9}{'Δppl':>8}{'KL(bf16‖q)':>12}"
          f"{'top-1':>9}{'prefill t/s':>13}{'decode t/s':>12}{'GB':>7}")
    print("-" * 104)
    for label in [REF] + QUANTS:
        r = results[label]
        d = f"{(r['ppl'] / ref_ppl - 1) * 100:+.2f}%"
        kl = f"{r['kl']:.5f}" if "kl" in r else "—"
        a = f"{r['top1_agree'] * 100:.2f}%" if "top1_agree" in r else "—"
        print(f"{label:<31}{r['disk_gb']:>6.1f}G{r['ppl']:>9.4f}{d:>8}{kl:>12}{a:>9}"
              f"{r.get('prefill_tps', 0):>13.0f}{r.get('decode_tps', 0):>12.1f}"
              f"{r.get('speed_peak_gb', 0):>7.1f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
