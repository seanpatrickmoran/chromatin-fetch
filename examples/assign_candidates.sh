#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${CFETCH_MODEL_DIR:?Set CFETCH_MODEL_DIR to a folder containing config.json and model.safetensors}"
: "${CFETCH_POSITIVE_DB:?Set CFETCH_POSITIVE_DB to the positive reference database}"
: "${CFETCH_NEGATIVE_DB:?Set CFETCH_NEGATIVE_DB to the negative reference database}"
: "${CFETCH_QUERY_DB:?Set CFETCH_QUERY_DB to the prepared candidate database}"

CFETCH_OUTPUT_DIR="${CFETCH_OUTPUT_DIR:-./cfetch_assignment_results}"
CFETCH_DEVICE="${CFETCH_DEVICE:-auto}"

cfetch-infer \
  --config-dir "${REPOSITORY_ROOT}/configs/paper" \
  --model "${CFETCH_MODEL_DIR}" \
  --reference-database "${CFETCH_POSITIVE_DB}" \
  --reference-database "${CFETCH_NEGATIVE_DB}" \
  --reference-labels "StrongPositive,Negatives" \
  --query-database "${CFETCH_QUERY_DB}" \
  --out-dir "${CFETCH_OUTPUT_DIR}" \
  --device "${CFETCH_DEVICE}"
