#!/usr/bin/env bash
# Training grid (data already built). Priority order so partial results are usable. Skips finished runs.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
[ -f data/sets/test_squad.jsonl ] || $PY src/make_sets.py || exit 1
T="nice -n 10 $PY src/train_eval.py"
$T --gen swap_mismatch --regime base
$T --gen swap_mismatch --regime pair      # our modification
$T --gen swap_mismatch --qblind           # question-blind probe
$T --gen swap_mismatch --regime placebo   # same question in both labels -> no question shortcut
$T --gen mix --regime base
$T --gen mix --regime pair
$T --gen llm_rewrite --regime base
$T --gen llm_rewrite --regime pair
$T --gen swap_typed --regime base
$T --gen swap_typed --regime pair
for r in swap_mismatch_base_s0 swap_mismatch_pair_s0 mix_base_s0 mix_pair_s0; do $PY src/attribution.py --run $r; done
$PY src/analyze.py
echo PIPELINE_DONE
