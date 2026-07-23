#!/usr/bin/env python3
"""Render per-quant model cards from card_template.md and upload them.

Renders straight from the template every time -- never patches an already-rendered
card. Patching rendered output is what previously produced duplicated headings
("## Sampling## Sampling") and a tier silently missing from the table.
Cards are validated before any upload.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Volumes/models/defiant-fable")
NAME = "Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX"
ORG = "pipenetwork"

TIERS = ["bf16", "8bit", "6bit", "5bit", "4bit", "3bit"]

DESC = {
    "bf16": "unquantized bfloat16, the full-precision reference",
    "8bit": "8-bit, measurably indistinguishable from bf16",
    "6bit": "6-bit, near-lossless",
    "5bit": "5-bit, the best quality-per-GB of the set",
    "4bit": "4-bit, the smallest tier we'd recommend for general use",
    "3bit": "3-bit, a tight-memory fallback — coherent but +33% perplexity",
}

SIZE_KEY = {"3bit": "__SZ3__", "4bit": "__SZ4__", "5bit": "__SZ5__",
            "6bit": "__SZ6__", "8bit": "__SZ8__", "bf16": "__SZ16__"}


def repo_size_gb(tag: str) -> float:
    d = ROOT / f"{NAME}-{tag}"
    return sum(f.stat().st_size for f in d.glob("*.safetensors")) / 1e9


def render(tag: str, sizes: dict, template: str) -> str:
    card = template.replace("__QUANT__", tag).replace("__QUANT_DESC__", DESC[tag])
    for t, key in SIZE_KEY.items():
        card = card.replace(key, f"{sizes[t]:.1f} GB")
    card = card.replace(
        "## License",
        "## Conversion tooling\n\nScripts, benchmark and the provenance proof: "
        "[github.com/PipeNetwork/defiant-fable-mlx]"
        "(https://github.com/PipeNetwork/defiant-fable-mlx)\n\n## License",
        1,
    )
    # mark this repo's own row in the comparison table
    card = card.replace(f"| [{tag}](https://huggingface.co/{ORG}/{NAME}-{tag}) |",
                        f"| **[{tag}](https://huggingface.co/{ORG}/{NAME}-{tag})** ← this repo |", 1)
    return card


def validate(tag: str, card: str) -> list:
    """Catch the failure modes that actually bit us, before anything is uploaded."""
    problems = []
    if "__" in card:
        problems += [f"unsubstituted placeholder: {m}" for m in set(re.findall(r"__[A-Z0-9_]+__", card))]
    for h in re.findall(r"^##+ .*$", card, re.M):
        if re.search(r"##+ .*##+ ", h):
            problems.append(f"duplicated heading: {h!r}")
    headings = re.findall(r"^(##+ .+)$", card, re.M)
    if len(headings) != len(set(headings)):
        dupes = {h for h in headings if headings.count(h) > 1}
        problems.append(f"repeated section(s): {sorted(dupes)}")
    for t in TIERS:  # every tier must appear in the comparison table
        if f"huggingface.co/{ORG}/{NAME}-{t}" not in card:
            problems.append(f"tier missing from card: {t}")
    if card.count("← this repo") != 1:
        problems.append(f"self-row marked {card.count('← this repo')} times, expected 1")
    if not card.startswith("---\n"):
        problems.append("missing YAML front matter")
    return problems


def main(tags):
    template = (ROOT / "card_template.md").read_text()
    sizes = {t: repo_size_gb(t) for t in TIERS}

    rendered = {}
    failed = False
    for tag in tags:
        card = render(tag, sizes, template)
        problems = validate(tag, card)
        if problems:
            failed = True
            for p in problems:
                print(f"!! {tag}: {p}")
        else:
            print(f"   {tag}: card OK ({len(card)} chars)")
        rendered[tag] = card
    if failed:
        print("refusing to upload; fix the template first")
        return 1

    for tag in tags:
        out = ROOT / f"{NAME}-{tag}" / "README.md"
        out.write_text(rendered[tag])
        repo = f"{ORG}/{NAME}-{tag}"
        r = subprocess.run(
            ["hf", "upload", repo, str(out), "README.md",
             "--commit-message", "Unified measured-quantization table across all tiers"],
            capture_output=True, text=True,
        )
        print(f"   {tag}: {'uploaded' if r.returncode == 0 else 'FAILED ' + r.stderr[-200:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or TIERS))
