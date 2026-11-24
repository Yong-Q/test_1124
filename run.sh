#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS="${XI_BLAS_THREADS:-1}" OMP_NUM_THREADS="${XI_BLAS_THREADS:-1}" MKL_NUM_THREADS="${XI_BLAS_THREADS:-1}" PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" "$root/run.py" "$@"
