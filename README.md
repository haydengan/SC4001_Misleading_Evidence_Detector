# Content or Fingerprint? Shortcut Learning in Neural Evidence Classifiers

SC4001 Neural Networks and Deep Learning, Group 167 (Gladys Woo Hui Min, Gan Yee Hong).

We fine-tune a small transformer that reads a (question, passage) pair and labels the passage **supporting**,
**misleading** or **irrelevant**, the kind of filter used to protect retrieval-augmented generation (RAG). Our question:
does it detect misleading *content*, or just the *traces of how the fake was made*?

- **Report:** `docs/SC4001_Group167_Report_v3.docx` (and `.pdf`)
- **Research design notes:** `RESEARCH_DESIGN.md`, `REPORT_NOTES.md`

## Main results (seed 0)

| Finding | Number |
|---|---|
| Standard baseline looks strong | AUROC 0.89 on its own fake type |
| ...but flags the same fake passage when the question is about an untouched part | 57% (bystander test) |
| Question-specific signal (2×2 DiD), baseline → pair training | 10 → 20 points |
| Same-type fakes, no reference → with a reference that contains the true answer | AUROC 0.57 → 0.70 |
| LLM-rewritten fakes, no reference → with reference | AUROC 0.53 → 0.64 |

## Setup

Python 3.11. Tested on an Apple M5 laptop (16 GB, MPS); CUDA and CPU are picked automatically.

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv torch transformers datasets accelerate scikit-learn pandas matplotlib spacy sentencepiece protobuf tqdm
uv pip install -p .venv "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

Datasets (SQuAD v1.1, NQ-Swap, FaithEval) and pretrained models download automatically from Hugging Face on first run.

## Reproduce everything

```bash
./run_all.sh
```

This runs data → LLM fakes → 10 single-passage runs → 4 cross-document runs → analysis → figures. Every step is
resumable and finished runs are skipped. Expect about 25–40 min per training run on a 16 GB laptop (about 8–10 h in total).

| Step | Command | Output |
|---|---|---|
| 1. SQuAD records, entity swaps, bystander/placebo | `src/build_data.py` | `data/records_*.jsonl` |
| 2. LLM rewrites (Qwen2.5-1.5B-Instruct) | `src/llm_fakes.py` | `data/llm_*.jsonl` |
| 3. Train/dev/test sets + NQ-Swap, FaithEval | `src/make_sets.py` | `data/sets/` |
| 4. Single-passage training (10 runs) + attribution | `./run_train.sh` | `runs/`, `results/attribution_*.json` |
| 5. Cross-document references and training (4 runs) | `./run_xdoc.sh` | `data/refs_*.json`, `runs/xdoc_*` |
| 6. Metrics and tables | `src/analyze.py`, `src/analyze_xdoc.py`, `src/calibrate.py` | `results/*.csv` |
| 7. Figures | `src/figures.py`, `src/fig_example.py` | `results/figures/` |

To train a single configuration:

```bash
.venv/bin/python src/train_eval.py --gen swap_mismatch --regime pair           # single passage
SETS_DIR=data/sets_xdoc .venv/bin/python src/train_eval.py --gen swap_typed --regime base \
    --tag xdoc_ --max_len 448 --accum 4                                         # with reference
```

`--gen` is one of `swap_mismatch`, `swap_typed`, `llm_rewrite`, `mix`; `--regime` is `base`, `pair` or `placebo`;
`--qblind` removes the question.

## View saved results (no retraining, under 1 second)

```bash
.venv/bin/python src/show_results.py training   # dev macro-F1 per epoch and time, every run
.venv/bin/python src/show_results.py main       # single-passage results table
.venv/bin/python src/show_results.py xdoc       # cross-document AUROC
.venv/bin/python src/show_results.py example    # two stored test examples
```

## Repository layout

| Path | Content |
|---|---|
| `src/` | All code (commented) |
| `results/` | Metric tables (`mean.csv`, `xdoc.csv`, `xdoc_auc.csv`, `calibrated.csv`), attribution, figures |
| `runs/<run>/` | `preds.jsonl` (test predictions), `metrics_dev.json`, `config.json`; model weights are not in git |
| `runs_plain_deberta/` | Earlier runs with plain DeBERTa-v3-small (report Appendix B) |
| `logs/` | Training and generation logs |
| `docs/` | Report, project handout, brainstorm |

Not in git (too large, rebuilt by `run_all.sh`): `data/`, model weights in `runs/*/model/`, `.venv/`.

## Credits

- **Pretrained models:** `cross-encoder/nli-deberta-v3-small` (DeBERTa-v3-small, He et al. 2023), Qwen2.5-1.5B-Instruct
  (Qwen Team 2024), spaCy `en_core_web_sm`.
- **Datasets:** SQuAD v1.1 (Rajpurkar et al. 2016), NQ-Swap (Longpre et al. 2021), FaithEval-counterfactual
  (Ming et al. 2025).
- All pipeline, diagnostic and analysis code was written for this project, with help from Anthropic Claude
  (see the AI Use Statement in the report).
