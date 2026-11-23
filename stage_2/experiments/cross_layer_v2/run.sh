#!/usr/bin/env bash
set -euo pipefail
cross_layer_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${cross_layer_dir}/../../env.sh"
cross_layer_command="${1:-report}"
if [[ $# -gt 0 ]]; then shift; fi
case "${cross_layer_command}" in
    formal)
        if [[ $# -eq 0 && -f "${cross_layer_dir}/formal/results/summary.json" ]]; then
            exec python "${cross_layer_dir}/formal/verify.py" --replay
        fi
        exec python "${cross_layer_dir}/formal/verify.py" "$@"
        ;;
    check)
        exec python "${cross_layer_dir}/verify_runtime.py" "$@"
        ;;
    pilot)
        exec python -u "${cross_layer_dir}/pilot.py" "$@"
        ;;
    report)
        exec python "${cross_layer_dir}/analyze_pilot.py" "$@"
        ;;
    all)
        if [[ $# -ne 2 || "$1" != --output ]]; then
            printf '%s\n' 'Usage: run.sh all --output NEW_DIRECTORY' >&2
            exit 2
        fi
        cross_layer_output="$2"
        if [[ -e "${cross_layer_output}" ]]; then
            printf '%s\n' 'Choose a new output directory; earlier measurements are preserved.' >&2
            exit 2
        fi
        mkdir -p -- "${cross_layer_output}"
        cross_layer_output="$(cd -- "${cross_layer_output}" && pwd)"
        python "${cross_layer_dir}/formal/verify.py" --output "${cross_layer_output}/formal"
        python "${cross_layer_dir}/verify_runtime.py" --output "${cross_layer_output}/runtime-check.json"
        python -u "${cross_layer_dir}/pilot.py" --output "${cross_layer_output}/pilot"
        python "${cross_layer_dir}/analyze_pilot.py" --input "${cross_layer_output}/pilot" \
            --runtime-check "${cross_layer_output}/runtime-check.json" \
            --report "${cross_layer_output}/开发验证报告.md"
        ;;
    *)
        printf '%s\n' 'Commands: formal [--output NEW_DIRECTORY], check, pilot, report, all --output NEW_DIRECTORY' >&2
        exit 2
        ;;
esac
