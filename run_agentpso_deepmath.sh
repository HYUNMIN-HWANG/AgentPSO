#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional: run with CONDA_ENV=AgentPSO ./run_agentpso_deepmath.sh
CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found, but CONDA_ENV=${CONDA_ENV} was requested." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND="${BACKEND:-openai}"
MODEL="${MODEL:-gpt-5.4-mini}"
RUN_NAME="${RUN_NAME:-deepmath_agentpso_train100_val100_test200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}}"
DEEPMATH_DATASET="${DEEPMATH_DATASET:-../../data/DeepMath-103K/data/train-00000-of-00010.parquet}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
DOWNLOAD_DEEPMATH="${DOWNLOAD_DEEPMATH:-1}"
DEEPMATH_SHARDS="${DEEPMATH_SHARDS:-0}"

TRAIN_POOL_SIZE="${TRAIN_POOL_SIZE:-100}"
VALIDATION_POOL_SIZE="${VALIDATION_POOL_SIZE:-100}"
TEST_LIMIT="${TEST_LIMIT:-200}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-20}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
TEST_MODE="${TEST_MODE:-personal_best}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-1}"

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

if [[ "${BACKEND}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for BACKEND=openai." >&2
  echo "Set OPENAI_API_KEY, or run with BACKEND=mock for a smoke test." >&2
  exit 1
fi

if [[ ! -f "${DEEPMATH_DATASET}" ]]; then
  if [[ "${DOWNLOAD_DEEPMATH}" != "1" ]]; then
    echo "DeepMath dataset not found: ${DEEPMATH_DATASET}" >&2
    echo "Set DOWNLOAD_DEEPMATH=1 or provide DEEPMATH_DATASET=/path/to/train-00000-of-00010.parquet." >&2
    exit 1
  fi
  echo "DeepMath dataset not found. Downloading shard(s): ${DEEPMATH_SHARDS}"
  "${PYTHON_BIN}" download_deepmath_103k.py --shards "${DEEPMATH_SHARDS}"
fi

if [[ ! -f "${DEEPMATH_DATASET}" ]]; then
  echo "DeepMath dataset is still missing after download: ${DEEPMATH_DATASET}" >&2
  exit 1
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.launch.log"
mkdir -p "${OUTPUT_ROOT}"
rm -f "${LOG_PATH}"

CMD=(
  "${PYTHON_BIN}" agent_pso.py
  --backend "${BACKEND}"
  --model "${MODEL}"
  --run-name "${RUN_NAME}"
  --output-root "${OUTPUT_ROOT}"
  --dataset-preset deepmath
  --train-dataset "${DEEPMATH_DATASET}"
  --test-dataset "${DEEPMATH_DATASET}"
  --train-pool-size "${TRAIN_POOL_SIZE}"
  --validation-pool-size "${VALIDATION_POOL_SIZE}"
  --test-limit "${TEST_LIMIT}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --validation-batch-size "${VALIDATION_BATCH_SIZE}"
  --num-iterations "${NUM_ITERATIONS}"
  --test-mode "${TEST_MODE}"
  --seed "${SEED}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running AgentPSO DeepMath train/test"
echo "run_dir=${RUN_DIR}"
echo "dataset=${DEEPMATH_DATASET}"
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"

status="${PIPESTATUS[0]}"
mkdir -p "${RUN_DIR}"
cp "${LOG_PATH}" "${RUN_DIR}/launch.log"
exit "${status}"
