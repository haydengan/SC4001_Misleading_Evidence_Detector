#!/usr/bin/env bash
# Full pipeline exactly as used for the report. Every step skips work that is already done, so re-running is safe.
# Tested on an Apple M5 laptop (16 GB, MPS); each training run takes ~25-40 min.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7

# 1) data: SQuAD records + rule-based fakes, then LLM rewrites (shortest passages) and one sampling retry pass
[ -f data/records_test.jsonl ] || $PY src/build_data.py
$PY src/llm_fakes.py --splits test --kinds fake,placebo --max_records test:936
$PY src/llm_fakes.py --splits dev,train --kinds fake --max_records dev:300,train:1500
$PY src/llm_fakes.py --retry --splits test --kinds fake,placebo
$PY src/llm_fakes.py --retry --splits dev,train --kinds fake

# 2) single-passage experiments (10 runs) + attribution + tables   (Section 4)
./run_train.sh

# 3) cross-document experiments (4 runs) + calibration             (Section 5)
./run_xdoc.sh
$PY src/analyze_xdoc.py
$PY src/calibrate.py

# 4) figures
$PY src/figures.py
