# SC4001 Project: Evidence classifier — content or fingerprint?

See `RESEARCH_DESIGN.md` for the research question, the novelty, and the experiment grid.

## Setup
```bash
uv venv -p 3.11 .venv
uv pip install -p .venv torch transformers datasets accelerate scikit-learn pandas matplotlib spacy sentencepiece protobuf tqdm
uv pip install -p .venv "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```
Tested on an Apple M5 (16 GB, MPS). It also runs on CUDA, where it picks the device automatically.

## Reproduce
```bash
./run_all.sh          # data -> LLM fakes -> sets -> all training runs -> attribution -> tables
```
Every step can be resumed, and training runs that already finished are skipped. Logs go to `logs/`, per-run outputs to `runs/<gen>_<regime>_s<seed>/`, and tables/figures to `results/`.

| Step | Script | Output |
|---|---|---|
| 1 | `src/build_data.py` | `data/records_{train,dev,test}.jsonl` (SQuAD, split by article) |
| 2 | `src/llm_fakes.py` | `data/llm_*.jsonl` (Qwen2.5-1.5B-Instruct rewrites) |
| 3 | `src/make_sets.py` | `data/sets/*.jsonl` |
| 4 | `src/train_eval.py` | `runs/*/preds.jsonl`, `metrics_dev.json` |
| 5 | `src/attribution.py`, `src/analyze.py` | `results/` |

## Credits (pretrained models / data)
DeBERTa-v3-small (Microsoft), Qwen2.5-1.5B-Instruct (Alibaba), spaCy en_core_web_sm, SQuAD v1.1 (Rajpurkar et al. 2016),
NQ-Swap (Longpre et al. 2021), FaithEval (Ming et al., Salesforce 2024).
