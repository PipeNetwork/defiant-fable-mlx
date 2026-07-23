#!/usr/bin/env python3
"""Compare quantizations of the same weights against their bf16 reference.

Every model here derives from nightmedia/Qwen3.5-9B-DS9-USS-Defiant, so bf16 is
exact ground truth and the quantization scheme is the only variable. Reports:

  * perplexity on a fixed wikitext-2 slice (lower is better)
  * KL(bf16 || quant) per token -- how far the quantized distribution drifts from
    the original; the sharpest quant-fidelity metric and harder to game than ppl
  * top-1 agreement with bf16's argmax
  * prefill / decode throughput and peak memory, measured separately

Evaluation is pairwise: bf16 plus exactly one quant resident at a time, fed
identical token ids. Holding every model at once overflows into a Metal command
buffer timeout, and buffering reference logprobs for the corpus would need ~32 GB.
bf16 is therefore recomputed once per quant, which costs ~50 s each and keeps the
comparison exact.
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
PATHS = {
    REF: ROOT / f"{NAME}-bf16",
    "pipenetwork 8bit (affine g64)": ROOT / f"{NAME}-8bit",
    "pipenetwork 6bit (affine g64)": ROOT / f"{NAME}-6bit",
    "pipenetwork 5bit (affine g64)": ROOT / f"{NAME}-5bit",
    "pipenetwork 4bit (affine g64)": ROOT / f"{NAME}-4bit",
    "nightmedia mxfp4 (g32)": ROOT / "mxfp4",
}
QUANTS = [k for k in PATHS if k != REF]


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


def eval_pair(ref_model, quant_model, x, y, n_seq, want_ref):
    """One pass over the corpus scoring the quant (and optionally bf16) ."""
    q = {"nll": 0.0, "kl": 0.0, "agree": 0, "n": 0}
    r = {"nll": 0.0, "n": 0}
    for i in range(n_seq):
        xi, yi = x[i : i + 1], y[i : i + 1]
        # Each logprob tensor is seq_len x vocab (~1 GB at fp32). Materializing the
        # two forwards and the KL reduction in one command buffer overruns the GPU
        # watchdog, so force them out in separate stages.
        ref_lp = logprobs_of(ref_model, xi)
        mx.eval(ref_lp)
        ref_top = ref_lp.argmax(axis=-1)
        mx.eval(ref_top)

        lp = logprobs_of(quant_model, xi)
        mx.eval(lp)

        tgt_q = mx.take_along_axis(lp, yi[..., None], axis=-1).squeeze(-1)
        mx.eval(tgt_q)
        kl = (mx.exp(ref_lp) * (ref_lp - lp)).sum(axis=-1)
        mx.eval(kl)
        agree = (lp.argmax(axis=-1) == ref_top).sum()
        mx.eval(agree)
        q["nll"] += float(-tgt_q.sum().item())
        q["kl"] += float(kl.sum().item())
        q["agree"] += int(agree.item())
        q["n"] += tgt_q.size

        if want_ref:
            tgt_r = mx.take_along_axis(ref_lp, yi[..., None], axis=-1).squeeze(-1)
            mx.eval(tgt_r)
            r["nll"] += float(-tgt_r.sum().item())
            r["n"] += tgt_r.size
            del tgt_r

        del ref_lp, ref_top, lp, tgt_q, kl, agree
        mx.clear_cache()
    return q, r


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
        mx.eval(next(it)[0])
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
    return {"prefill_tps": best_prefill, "decode_tps": best_decode, "peak_gb": peak}


def component_sizes(path):
    """Split on-disk weights into language model vs vision tower."""
    out = {"language": 0, "vision": 0}
    for f in sorted(Path(path).glob("*.safetensors")):
        for k, v in mx.load(str(f)).items():
            grp = "vision" if k.startswith("vision_tower") else "language"
            out[grp] += v.size * v.dtype.size
    return {k: v / 1e9 for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()

    from mlx_lm.utils import load

    for label, p in PATHS.items():
        if not p.exists():
            raise SystemExit(f"missing {label}: {p}")

    ref_model, ref_tok = load(str(PATHS[REF]))
    x, y, n_seq = build_corpus(ref_tok, args.tokens, args.seq_len)
    print(f"corpus: {n_seq} x {args.seq_len} = {n_seq * args.seq_len} tokens", flush=True)

    results = {}
    for j, label in enumerate(QUANTS):
        print(f"\n== {label}", flush=True)
        qm, qtok = load(str(PATHS[label]))
        assert len(qtok.vocab) == len(ref_tok.vocab), f"vocab mismatch for {label}"
        q, r = eval_pair(ref_model, qm, x, y, n_seq, want_ref=(j == 0))
        results[label] = {
            "ppl": float(np.exp(q["nll"] / q["n"])),
            "kl": q["kl"] / q["n"],
            "top1_agree": q["agree"] / q["n"],
        }
        if j == 0:
            results[REF] = {"ppl": float(np.exp(r["nll"] / r["n"]))}
        print(f"   ppl={results[label]['ppl']:.4f}  KL={results[label]['kl']:.5f}"
              f"  top1={results[label]['top1_agree'] * 100:.2f}%", flush=True)
        del qm
        gc.collect()
        mx.clear_cache()

    del ref_model
    gc.collect()
    mx.clear_cache()

    if not args.skip_speed:
        print("\nmeasuring speed (one model at a time) ...", flush=True)
        for label, p in PATHS.items():
            results[label].update(measure_speed(p))
            print(f"  {label}: {results[label]['decode_tps']:.1f} tok/s decode", flush=True)

    for label, p in PATHS.items():
        results[label]["sizes"] = component_sizes(p)

    (ROOT / "bench_results.json").write_text(json.dumps(results, indent=2))

    ref_ppl = results[REF]["ppl"]
    order = [REF] + QUANTS
    print("\n" + "=" * 108)
    print(f"{'model':<31}{'lang':>7}{'ppl':>9}{'Δppl':>9}{'KL(bf16‖q)':>12}"
          f"{'top-1':>9}{'prefill':>10}{'decode':>9}{'GB':>7}")
    print("-" * 108)
    for label in order:
        r = results[label]
        d = f"{(r['ppl'] / ref_ppl - 1) * 100:+.2f}%"
        kl = f"{r['kl']:.5f}" if "kl" in r else "—"
        a = f"{r['top1_agree'] * 100:.2f}%" if "top1_agree" in r else "—"
        print(f"{label:<31}{r['sizes']['language']:>6.2f}G{r['ppl']:>9.4f}{d:>9}{kl:>12}{a:>9}"
              f"{r.get('prefill_tps', 0):>10.0f}{r.get('decode_tps', 0):>9.1f}"
              f"{r.get('peak_gb', 0):>7.1f}")
    print("=" * 108)


if __name__ == "__main__":
    main()
