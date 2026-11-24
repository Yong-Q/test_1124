#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:?Usage: smoke.sh OUTPUT_DIR [cpu|cuda]}"
device="${2:-cpu}"
mkdir -p "$output"
bash "$root/run.sh" validate > "$output/reference.json"
for variant in fft-excess fft-intrinsic sparse-excess cnn-single-excess cnn-single-intrinsic cnn-dual-excess stable-base stable-bounded; do
  bash "$root/run.sh" train "$variant" --case ch4 --device "$device" --epochs 1 --memory-gib 8 --output "$output/$variant" > "$output/$variant.stdout"
  printf '%s %s passed\n' "$variant" "$device"
done
