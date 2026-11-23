#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${experiment_dir}/../../env.sh"
if [[ $# -eq 0 ]]; then
    set -- --report --output "${experiment_dir}/results/replay"
fi
exec python "${experiment_dir}/replay.py" "$@"
