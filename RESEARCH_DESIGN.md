# Does a trained evidence classifier detect misleading passages, or just how they were made?

SC4001 Group Project · Track F (Reliable RAG) · Deadline 16 Nov 2026

## One-line pitch
We train a small cross-encoder to label (question, passage) as **supporting / misleading / irrelevant**, then
use *controlled counterfactual tests* to separate two hypotheses:
- **H-content**: it learns "this passage gives a wrong answer *to this question*"
- **H-fingerprint**: it learns "this passage was edited by generator X" (regardless of the question)

## Why this is novel (what we contribute beyond "train a classifier")
1. **Controlled fabrication.** All in-domain fakes come from the *same* SQuAD passages and (for two of the
   three generators) the *same* fake answer, so differences in detection come from the fabrication process
   alone:
   | Generator | What it does | Expected fingerprint |
   |---|---|---|
   | `swap_mismatch` | answer → random answer of a *different* type | crude: type clash ("born in Microsoft") |
   | `swap_typed` | answer → same-type entity (NQ-Swap style) | subtle: surface edit, leftover inconsistencies |
   | `llm_rewrite` | Qwen2.5-1.5B rewrites passage to state the *same* typed fake answer | fluent, globally consistent, paraphrase drift |
   Plus two **external** generators made by other people on other data: NQ-Swap, FaithEval-counterfactual.
2. **Two fingerprint controls (our main diagnostic):**
   - **Bystander test** – the *identical* fake passage, paired with a *different* SQuAD question about the same
     paragraph whose answer was untouched → correct label is *supporting*. Text is held fixed; only the
     question changes. A fingerprint detector flags it; a content detector does not.
   - **Placebo test** – the same generator applied to a *non-answer* entity → passage carries the edit
     fingerprint but still supports the question.
   - **Content Sensitivity** `CS = TPR(fake) − FAR(bystander)`; CS≈0 ⇒ pure fingerprint detection.
3. **Question-blind probe.** Same model, question removed. Misleading-vs-original accuracy without the question
   can only come from passage artefacts (or world knowledge) → upper bound on the fingerprint signal.
4. **Modification: question-conditioned counterfactual pairing.** In training, half the *supporting* examples are
   bystanders (same fake passage, other question). The edit fingerprint now appears under both labels, so the
   only way to fit the data is to check whether the edit hits the span the question asks about. Same data size
   as baseline. Ablation: placebo-pairing instead. Comparison: multi-generator mixture (the standard remedy in
   AI-image detection).
5. **Attribution analysis.** Gradient×input mass on the edited span for fakes vs bystanders.

## Data
- SQuAD v1.1 train → split *by article* into train (90%) / dev (10%); SQuAD validation → test (unseen articles).
- Questions kept only if answer is a typed named entity / number / date with a placebo candidate and a bystander.
- Irrelevant: 50% passage from another article (easy), 50% another paragraph of the *same* article (hard).
- Labels: 0 supporting, 1 misleading, 2 irrelevant.

## Experiments
| ID | Train generator | Regime | Purpose |
|---|---|---|---|
| E1 | each of 3 + mixture | base | cross-generator matrix, fingerprint diagnostics |
| E2 | each of 3 + mixture | qblind | how much is solvable without the question |
| E3 | each of 3 + mixture | pair (ours) | does pairing raise CS / generalisation? |
| E4 | swap_typed | placebo | ablation of the pairing source |
13 runs, seed 0 (extra seeds only for E1 vs E3 if time allows). Backbone: `microsoft/deberta-v3-small`.

**Lean-scale note (compute budget: one 16 GB laptop).** Train = 1,500 shortest-passage train questions with a
valid LLM fake; dev = 300; test = 936 SQuAD-val questions with valid LLM fake + placebo (also the shorter
passages, so train/test length profiles match); external = 500 NQ-Swap + 500 FaithEval. All generators share
the same questions, so comparisons stay controlled. Limitation to report: shorter passages only.

## Layout
- `src/build_data.py` – SQuAD → typed records, rule fakes, placebos, bystanders, LLM job list
- `src/llm_fakes.py` – runs Qwen2.5-1.5B-Instruct for `llm_rewrite` fakes/placebos
- `src/make_sets.py` – assembles train/dev/test jsonl per generator + external tests
- `src/train_eval.py` – trains one config, evaluates on every test set
- `src/attribution.py`, `src/analyze.py` – figures/tables
- `docs/` – brief, brainstorm, old plan
