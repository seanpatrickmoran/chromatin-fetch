#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$TEST_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$TEST_DIR/outputs"}

"$PYTHON" "$ROOT_DIR/sample_coordinates.py" \
  --regions "$TEST_DIR/chr6_region.yaml" \
  --out "$OUTPUT_ROOT/chr6_statistical" \
  --tile sampling \
  --resolutions 5kb \
  --minrange 50kb \
  --maxrange 2mb \
  --samples-per-start 4 "$@"

"$PYTHON" "$ROOT_DIR/sample_coordinates.py" \
  --regions "$TEST_DIR/chr6_region.yaml" \
  --out "$OUTPUT_ROOT/chr6_gumbel" \
  --tile sampling \
  --resolutions 5kb \
  --minrange 50kb \
  --maxrange 2mb \
  --num-samples 20000 \
  --decay gumbel \
  --decay-start 1mb \
  --decay-scale 200kb "$@"

"$PYTHON" "$ROOT_DIR/sample_coordinates.py" \
  --regions "$TEST_DIR/chr6_region.yaml" \
  --out "$OUTPUT_ROOT/chr6_powerlaw" \
  --tile sampling \
  --resolutions 5kb \
  --minrange 50kb \
  --maxrange 2mb \
  --num-samples 20000 \
  --decay powerlaw \
  --decay-start 500kb \
  --beta 2 "$@"

"$PYTHON" "$ROOT_DIR/sample_coordinates.py" \
  --regions "$TEST_DIR/chr6_region.yaml" \
  --out "$OUTPUT_ROOT/chr6_full" \
  --tile force_full \
  --resolutions 5kb \
  --minrange 50kb \
  --maxrange 2mb "$@"
