#!/bin/bash
# Convert nightmedia/Qwen3.5-9B-DS9-USS-Defiant -> MLX quants (vision-enabled)
#
# mlx-vlm's Qwen3_5 sanitize() adds +1.0 to RMSNorm weights on every load, but
# its converter bakes that shift into the saved file -- so a freshly converted
# repo double-shifts when mlx-vlm loads it back. fix_norms.py rewrites the norm
# and conv1d tensors in raw-HF convention so mlx-vlm AND mlx-lm each apply the
# shift exactly once. verify.py refuses to let a broken repo through.
set -euo pipefail

ROOT=/Volumes/models/defiant-fable
SRC=$ROOT/src
NAME=Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-MLX

cd "$ROOT"

build() {
  local tag=$1; shift
  local out="$ROOT/$NAME-$tag"
  if [ -f "$out/.verified" ]; then
    echo "== $tag already built and verified, skipping"
    return
  fi
  echo "== building $tag"
  rm -rf "$out"
  python3 -m mlx_vlm convert --hf-path "$SRC" --mlx-path "$out" "$@" 2>&1 | tail -4
  python3 fix_norms.py "$out" "$SRC" 2>&1 | tail -6
  python3 verify.py "$out" 2>&1 | tail -6
  touch "$out/.verified"
  du -sh "$out"
}

build bf16
build 8bit -q --q-bits 8 --q-group-size 64
build 6bit -q --q-bits 6 --q-group-size 64
build 5bit -q --q-bits 5 --q-group-size 64
build 4bit -q --q-bits 4 --q-group-size 64
build 3bit -q --q-bits 3 --q-group-size 64
build 2bit -q --q-bits 2 --q-group-size 64

echo "== all done"
du -sh "$ROOT/$NAME"-*
