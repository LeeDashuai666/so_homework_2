#!/usr/bin/env bash
# Source this file: source stage_2/env.sh
_stage2_env_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${_stage2_env_root}/.venv/bin/activate" ]]; then
    printf '%s\n' 'Run: python3 stage_2/tools/setup_env.py' >&2
    unset _stage2_env_root
    return 1 2>/dev/null || exit 1
fi
source "${_stage2_env_root}/.venv/bin/activate"
export STAGE2_ROOT="${_stage2_env_root}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS=2
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${STAGE2_ROOT}/.cache/matplotlib"
unset _stage2_env_root
