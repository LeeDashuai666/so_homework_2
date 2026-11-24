#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../../env.sh"
mode="${1:-report}"
output="${2:-$here/results/main}"
policy_model="${STAGE2_POLICY_MODEL:-$HOME/infer-lab/models/Qwen3-0.6B}"
case "$mode" in
  all)
    python -u "$here/bench.py" dev "$output"
    python -u "$here/select_ai.py" "$output" --model "$policy_model"
    python -u "$here/bench.py" eval "$output"
    python "$here/analyze.py" "$output"
    ;;
  dev|eval) python -u "$here/bench.py" "$mode" "$output" ;;
  ai) python -u "$here/select_ai.py" "$output" --model "$policy_model" ;;
  report) python "$here/analyze.py" "$output" ;;
  *) printf '%s\n' 'Usage: run.sh {all|dev|ai|eval|report} [results-directory]' >&2; exit 2 ;;
esac
