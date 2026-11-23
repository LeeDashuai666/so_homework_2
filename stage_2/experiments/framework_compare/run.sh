#!/usr/bin/env bash
set -euo pipefail
framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${framework_dir}/../../env.sh"
framework_command="${1:-report}"
if [[ $# -gt 0 ]]; then shift; fi
case "${framework_command}" in
    prepare)
        exec python "${framework_dir}/prepare_data.py" "$@"
        ;;
    development)
        exec python -u "${framework_dir}/execute.py" development "$@"
        ;;
    experiments)
        exec python -u "${framework_dir}/execute.py" formal "$@"
        ;;
    report)
        exec python "${framework_dir}/../../tools/analyze_framework_compare.py" "$@"
        ;;
    all)
        python "${framework_dir}/prepare_data.py"
        python -u "${framework_dir}/execute.py" all
        python "${framework_dir}/../../tools/analyze_framework_compare.py" "$@"
        ;;
    *)
        printf '%s\n' 'Commands: prepare, development, experiments, report, all' >&2
        exit 2
        ;;
esac
