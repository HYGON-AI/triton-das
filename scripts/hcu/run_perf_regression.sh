#!/bin/bash
#
# Nightly vs local Triton perf compare.
#
#   ./scripts/hcu/run_perf_regression.sh
#   ./scripts/hcu/run_perf_regression.sh -g 2
#   ./scripts/hcu/run_perf_regression.sh --aiter-dir /path/to/aiter --nightly-whl /path/to.whl

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_perf_regression.sh [OPTIONS]

Nightly vs local compare only.
  - always git-sync + setup.py develop aiter (unless --aiter-dir / $AITER_ROOT)
  - always resolve + download matching nightly whl (unless --nightly-whl)
  - reinstall local Triton and write compare.md

Options:
  -g, --gpu ID              Physical GPU id (else auto-select)
  -o, --output PATH         Output directory
      --aiter-dir PATH      Use this aiter as-is (skip sync/install)
      --nightly-whl PATH    Use this .whl (skip download)
      --operators OPS...    Subset of operators
      --resolve-nightly-only
      --fail-on-error
      --json PATH
      --aiter-ref / --nightly-index / --llvm-root
  -h, --help

Examples:
  ./scripts/hcu/run_perf_regression.sh
  ./scripts/hcu/run_perf_regression.sh --aiter-dir /xukang/code/aiter
  ./scripts/hcu/run_perf_regression.sh --aiter-dir /xukang/code/aiter \
      --nightly-whl /tmp/triton-xxx.whl
EOF
}

CUR_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CUR_PATH}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -g|--gpu|-o|--output|--aiter-dir|--aiter-ref|--nightly-whl|--nightly-index|--nightly-version-prefix|--llvm-root|--json)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      args+=("$1" "$2")
      shift 2
      ;;
    --operators)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      args+=(--operators)
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        args+=("$1")
        shift
      done
      ;;
    --fail-on-error|--resolve-nightly-only)
      args+=("$1")
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"
exec "${PYTHON}" "${CUR_PATH}/perf_report.py" "${args[@]}"
