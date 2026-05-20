#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="/workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2"
REPO_ROOT="/workspace/alpamayo_vlm"
REPLAY_SCRIPT="${REPO_ROOT}/scripts/run_live_chunk_udp_replay.py"
ENGINE_DIR="/workspace/models/alpamayo_runtime/engines/alpa1.5"
MULTIMODAL_ENGINE_DIR="/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"
FM_ENGINE="/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan"

ORACLE_NAV_TAG="20260428"
RUN_TAG="20260428"
VIEWER_HOST="0.0.0.0"
VIEWER_PORT=8780
REPLAY_MODE="latest_only"

SKIP_UDP=1
OPEN_VIEWER=0
ALPAMAYO_NAV_CFG=1
ALPAMAYO_FM_USE_PREFILL_KV=0
REQUEST_LIMIT=""
TARGET_OFFSET_S=""
REQUEST_STRIDE=""
EXTRA_ARGS=()
CHUNKS=()

usage() {
  cat <<'EOF'
Usage:
  run_live_chunk_oracle_nav_batch.sh [options]

Default behavior:
  - Runs chunk0000 ~ chunk0009 sequentially
  - Uses the prebuilt oracle-nav request banks
  - Uses FP8 FM engine
  - Uses latest_only replay mode
  - Uses --skip-udp

Options:
  --chunks "0 1 2"         Space-separated chunk ids to run
  --start N                Start chunk id for range mode
  --end N                  End chunk id for range mode
  --dataset-root PATH
  --oracle-nav-tag TAG     Request-bank tag, default: 20260428
  --run-tag TAG            Work-root tag, default: 20260428
  --viewer-host HOST
  --viewer-port PORT
  --replay-mode MODE       sequential | latest_only
  --request-limit N
  --request-stride N
  --target-offset-s S
  --open-viewer            Open browser for each chunk run
  --no-skip-udp            Send UDP instead of local-only replay
  --no-nav-cfg             Disable --alpamayo-nav-cfg
  --alpamayo-fm-use-prefill-kv
                           Feed backbone prefill KV directly into FM
  --engine-dir PATH
  --multimodal-engine-dir PATH
  --fm-engine PATH
  --extra-arg ARG          Extra arg passed through to run_live_chunk_udp_replay.py
  -h, --help

Examples:
  bash scripts/run_live_chunk_oracle_nav_batch.sh
  bash scripts/run_live_chunk_oracle_nav_batch.sh --chunks "6 7 8"
  bash scripts/run_live_chunk_oracle_nav_batch.sh --start 3 --end 5 --viewer-port 8781
EOF
}

START_CHUNK=0
END_CHUNK=9
CUSTOM_RANGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks)
      read -r -a CHUNKS <<< "${2:-}"
      shift 2
      ;;
    --start)
      START_CHUNK="${2:?missing value for --start}"
      CUSTOM_RANGE=1
      shift 2
      ;;
    --end)
      END_CHUNK="${2:?missing value for --end}"
      CUSTOM_RANGE=1
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="${2:?missing value for --dataset-root}"
      shift 2
      ;;
    --oracle-nav-tag)
      ORACLE_NAV_TAG="${2:?missing value for --oracle-nav-tag}"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="${2:?missing value for --run-tag}"
      shift 2
      ;;
    --viewer-host)
      VIEWER_HOST="${2:?missing value for --viewer-host}"
      shift 2
      ;;
    --viewer-port)
      VIEWER_PORT="${2:?missing value for --viewer-port}"
      shift 2
      ;;
    --replay-mode)
      REPLAY_MODE="${2:?missing value for --replay-mode}"
      shift 2
      ;;
    --request-limit)
      REQUEST_LIMIT="${2:?missing value for --request-limit}"
      shift 2
      ;;
    --request-stride)
      REQUEST_STRIDE="${2:?missing value for --request-stride}"
      shift 2
      ;;
    --target-offset-s)
      TARGET_OFFSET_S="${2:?missing value for --target-offset-s}"
      shift 2
      ;;
    --open-viewer)
      OPEN_VIEWER=1
      shift
      ;;
    --no-skip-udp)
      SKIP_UDP=0
      shift
      ;;
    --no-nav-cfg)
      ALPAMAYO_NAV_CFG=0
      shift
      ;;
    --alpamayo-fm-use-prefill-kv)
      ALPAMAYO_FM_USE_PREFILL_KV=1
      shift
      ;;
    --engine-dir)
      ENGINE_DIR="${2:?missing value for --engine-dir}"
      shift 2
      ;;
    --multimodal-engine-dir)
      MULTIMODAL_ENGINE_DIR="${2:?missing value for --multimodal-engine-dir}"
      shift 2
      ;;
    --fm-engine)
      FM_ENGINE="${2:?missing value for --fm-engine}"
      shift 2
      ;;
    --extra-arg)
      EXTRA_ARGS+=("${2:?missing value for --extra-arg}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#CHUNKS[@]} -eq 0 ]]; then
  for ((chunk=START_CHUNK; chunk<=END_CHUNK; chunk++)); do
    CHUNKS+=("${chunk}")
  done
fi

if [[ ${#CHUNKS[@]} -eq 0 ]]; then
  echo "No chunks selected." >&2
  exit 1
fi

if [[ ! -f "${REPLAY_SCRIPT}" ]]; then
  echo "Missing replay script: ${REPLAY_SCRIPT}" >&2
  exit 1
fi

echo "[oracle-nav-batch] dataset_root=${DATASET_ROOT}"
echo "[oracle-nav-batch] oracle_nav_tag=${ORACLE_NAV_TAG} run_tag=${RUN_TAG}"
echo "[oracle-nav-batch] chunks=${CHUNKS[*]}"
echo "[oracle-nav-batch] replay_mode=${REPLAY_MODE} viewer_port=${VIEWER_PORT} skip_udp=${SKIP_UDP} nav_cfg=${ALPAMAYO_NAV_CFG} prefill_kv=${ALPAMAYO_FM_USE_PREFILL_KV}"

for chunk_id in "${CHUNKS[@]}"; do
  printf -v pad "%04d" "${chunk_id}"
  chunk_tag="chunk${pad}"
  request_bank_root="${REPO_ROOT}/output/live_chunk_udp_replay_${chunk_tag}_oracle_nav_${ORACLE_NAV_TAG}/${chunk_tag}/request_bank"
  work_root="${REPO_ROOT}/output/live_chunk_udp_replay_${chunk_tag}_oracle_nav_fp8_${RUN_TAG}"

  if [[ ! -d "${request_bank_root}" ]]; then
    echo "[oracle-nav-batch] missing request bank for ${chunk_tag}: ${request_bank_root}" >&2
    exit 1
  fi

  cmd=(
    python3 "${REPLAY_SCRIPT}"
    --dataset-root "${DATASET_ROOT}"
    --chunk-id "${chunk_id}"
    --work-root "${work_root}"
    --request-bank-root "${request_bank_root}"
    --replay-mode "${REPLAY_MODE}"
    --engine-dir "${ENGINE_DIR}"
    --multimodal-engine-dir "${MULTIMODAL_ENGINE_DIR}"
    --fm-engine "${FM_ENGINE}"
    --viewer-host "${VIEWER_HOST}"
    --viewer-port "${VIEWER_PORT}"
  )

  if [[ "${ALPAMAYO_NAV_CFG}" -eq 1 ]]; then
    cmd+=(--alpamayo-nav-cfg)
  fi
  if [[ "${ALPAMAYO_FM_USE_PREFILL_KV}" -eq 1 ]]; then
    cmd+=(--alpamayo-fm-use-prefill-kv)
  fi
  if [[ "${OPEN_VIEWER}" -eq 1 ]]; then
    cmd+=(--open-viewer)
  fi
  if [[ "${SKIP_UDP}" -eq 1 ]]; then
    cmd+=(--skip-udp)
  fi
  if [[ -n "${REQUEST_LIMIT}" ]]; then
    cmd+=(--request-limit "${REQUEST_LIMIT}")
  fi
  if [[ -n "${REQUEST_STRIDE}" ]]; then
    cmd+=(--request-stride "${REQUEST_STRIDE}")
  fi
  if [[ -n "${TARGET_OFFSET_S}" ]]; then
    cmd+=(--target-offset-s "${TARGET_OFFSET_S}")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo
  echo "[oracle-nav-batch] running ${chunk_tag}"
  printf '[oracle-nav-batch] cmd:'; printf ' %q' "${cmd[@]}"; printf '\n'
  "${cmd[@]}"
done

echo
echo "[oracle-nav-batch] completed chunks: ${CHUNKS[*]}"
