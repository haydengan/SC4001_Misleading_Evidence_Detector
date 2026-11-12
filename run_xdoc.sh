#!/usr/bin/env bash
# Cross-document consistency experiment (question + reference snippet from other paragraphs, candidate passage).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
[ -f data/refs_test.json ] || $PY src/xdoc_refs.py
[ -f data/sets_xdoc/test_squad.jsonl ] || (cd src && ../$PY make_sets_xdoc.py)
for g in swap_typed swap_mismatch llm_rewrite; do
  SETS_DIR=data/sets_xdoc nice -n 10 $PY src/train_eval.py --gen $g --regime base --tag xdoc_ --max_len 448 --accum 4
done
echo XDOC_DONE
