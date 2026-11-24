#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
manifest="${1:?Provide the existing 152-probe training_split.json}"
output="${2:-runs/c2h4_low_density_3d}"
exec bash "$root/run.sh" perturb --manifest "$manifest" \
  --case C2H4-source239-C2H4-PRL-T430.3-r00-0.000291 \
  --count "${COUNT:-300}" --workers "${WORKERS:-1}" --output "$output"
