#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
nice -n 15 .venv/bin/python src/llm_fakes.py --retry --splits test --kinds fake,placebo || exit 1
nice -n 15 .venv/bin/python src/llm_fakes.py --retry --splits dev,train --kinds fake || exit 1
./run_train.sh
