#!/usr/bin/env bash
# Lean pipeline. Every step is resumable and finished training runs are skipped, so re-running is safe.
# Each training run is ~10-15 min on an Apple M-series laptop; run them one at a time with run_one.sh if preferred.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6  # keep the laptop usable

# 1) data
[ -f data/records_test.jsonl ] || $PY src/build_data.py
$PY src/llm_fakes.py --splits test --kinds fake,placebo --max_records test:936  # test: fakes + placebos (936 shortest)
$PY src/llm_fakes.py                                                        # dev/train: fakes only (shortest passages)
$PY src/make_sets.py

# 2) training: 4 train generators x {base, pair, qblind} + placebo ablation on swap_typed = 13 runs
for g in swap_mismatch swap_typed llm_rewrite mix; do
  for r in base pair; do nice -n 10 $PY src/train_eval.py --gen $g --regime $r; done
  nice -n 10 $PY src/train_eval.py --gen $g --qblind
done
nice -n 10 $PY src/train_eval.py --gen swap_typed --regime placebo

# 3) analysis
for r in swap_typed_base_s0 swap_typed_pair_s0 mix_base_s0 mix_pair_s0; do $PY src/attribution.py --run $r; done
$PY src/analyze.py
