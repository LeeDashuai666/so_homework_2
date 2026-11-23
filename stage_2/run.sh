#!/usr/bin/env bash
set -euo pipefail
stage2_run_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stage2_action="${1:-smoke-vlm}"
if [[ $# -gt 0 ]]; then shift; fi
case "${stage2_action}" in
    setup)
        exec python3 "${stage2_run_root}/tools/setup_env.py" "$@"
        ;;
    prepare-data|calibrate|experiments|unit|verify-results|report|test|rerun|package|status|all)
        source "${stage2_run_root}/env.sh"
        case "${stage2_action}" in
            prepare-data)
                python "${stage2_run_root}/tools/prepare_vlm_assets.py" --http
                python "${stage2_run_root}/tools/download_model_http.py" --lock "${stage2_run_root}/configs/experiment-assets.lock.json" --section dataset --workers 8
                exec python "${stage2_run_root}/scripts/prepare_experiment.py" "$@" ;;
            calibrate) exec python -u "${stage2_run_root}/scripts/pipeline.py" calibrate "$@" ;;
            experiments) exec python -u "${stage2_run_root}/scripts/pipeline.py" full "$@" ;;
            unit) exec python -u "${stage2_run_root}/scripts/experiment.py" unit "$@" ;;
            verify-results) exec python "${stage2_run_root}/scripts/verify_results.py" "$@" ;;
            report) exec python "${stage2_run_root}/scripts/analyze.py" "$@" ;;
            test)
                python -m pytest -q "${stage2_run_root}/tests"
                exec python "${stage2_run_root}/scripts/formally_verify.py" "$@" ;;
            rerun) exec python "${stage2_run_root}/scripts/replay.py" "$@" ;;
            package) exec python "${stage2_run_root}/scripts/package.py" "$@" ;;
            status) exec python "${stage2_run_root}/scripts/status.py" "$@" ;;
            all)
                if [[ ! -f "${stage2_run_root}/models/qwen3-vl-2b-instruct/model.safetensors" ]]; then
                    python "${stage2_run_root}/tools/prepare_vlm_assets.py" --http --model-only
                fi
                if [[ ! -f "${stage2_run_root}/data/processed/docvqa/manifest.json" ]]; then
                    bash "${stage2_run_root}/run.sh" prepare-data
                fi
                python -u "${stage2_run_root}/scripts/pipeline.py" full "$@"
                python -m pytest -q "${stage2_run_root}/tests"
                python "${stage2_run_root}/scripts/formally_verify.py"
                python "${stage2_run_root}/scripts/analyze.py"
                exec python "${stage2_run_root}/scripts/package.py" ;;
        esac
        ;;
    prepare|prepare-vlm|smoke|smoke-vlm|ask|ask-vlm|prepare-rerank|smoke-rerank|probe)
        source "${stage2_run_root}/env.sh"
        case "${stage2_action}" in
            prepare|prepare-vlm) exec python "${stage2_run_root}/tools/prepare_vlm_assets.py" "$@" ;;
            smoke|smoke-vlm) exec python "${stage2_run_root}/tools/smoke_vlm.py" "$@" ;;
            ask|ask-vlm) exec python "${stage2_run_root}/tools/ask_vlm.py" "$@" ;;
            prepare-rerank) exec python "${stage2_run_root}/tools/prepare_assets.py" "$@" ;;
            smoke-rerank) exec python "${stage2_run_root}/tools/smoke_env.py" "$@" ;;
            probe) exec python "${stage2_run_root}/tools/probe_env.py" --cuda-smoke "$@" ;;
        esac
        ;;
    *)
        printf '%s\n' 'Usage: ./stage_2/run.sh {setup|prepare-data|calibrate|experiments|all|unit|status|verify-results|report|test|rerun|package|smoke-vlm|ask-vlm|probe}' >&2
        exit 2
        ;;
esac
