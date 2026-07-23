#!/usr/bin/env python3
"""Write per-quant model cards and upload each repo to the pipenetwork org."""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Volumes/models/defiant-fable")
NAME = "Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX"
ORG = "pipenetwork"

DESC = {
    "bf16": "unquantized bfloat16, the full-precision reference",
    "8bit": "8-bit, effectively lossless against bf16",
    "6bit": "6-bit, the quality/size midpoint",
    "4bit": "4-bit, the smallest tier and the usual default",
}


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.glob("*.safetensors")) / 1e9


def main(tags):
    template = (ROOT / "card_template.md").read_text()
    sizes = {t: dir_size_gb(ROOT / f"{NAME}-{t}") for t in DESC if (ROOT / f"{NAME}-{t}").exists()}

    for tag in tags:
        out = ROOT / f"{NAME}-{tag}"
        if not (out / ".verified").exists():
            print(f"!! {tag} has no .verified marker -- refusing to publish")
            continue
        card = template.replace("__QUANT__", tag).replace("__QUANT_DESC__", DESC[tag])
        for t, key in (("4bit", "__SZ4__"), ("6bit", "__SZ6__"), ("8bit", "__SZ8__"), ("bf16", "__SZ16__")):
            card = card.replace(key, f"{sizes[t]:.1f} GB" if t in sizes else "-")
        (out / "README.md").write_text(card)

        (out / ".verified").unlink()  # build marker, not part of the release
        repo = f"{ORG}/{NAME}-{tag}"
        print(f"== uploading {repo} ({sizes.get(tag, 0):.1f} GB)")
        r = subprocess.run(
            ["hf", "upload", repo, str(out), ".", "--repo-type", "model",
             "--commit-message", "MLX conversion of Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic"],
            cwd=ROOT,
        )
        if r.returncode != 0:
            print(f"!! upload failed for {repo}")
            sys.exit(1)
        print(f"   https://huggingface.co/{repo}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["4bit", "6bit", "8bit", "bf16"])
