#!/usr/bin/env bash
set -euo pipefail
stage2_perf_script="$(readlink -f -- "${BASH_SOURCE[0]}")"
stage2_perf_root="$(cd -- "$(dirname -- "${stage2_perf_script}")/.." && pwd)"
stage2_perf_tools="${stage2_perf_root}/../stage_1/.tools/root"
stage2_perf_binary="${stage2_perf_root}/../stage_1/.tools/bin/perf"
if [[ ! -x "${stage2_perf_binary}" ]]; then
    printf '%s\n' 'The reused stage_1 perf binary is missing.' >&2
    exit 1
fi
export LD_LIBRARY_PATH="${stage2_perf_tools}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${stage2_perf_binary}" "$@"
