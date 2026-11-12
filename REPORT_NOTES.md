# Report Notes: Does a trained evidence classifier detect misleading passages, or just how they were made?

Working notes to lift into the SC4001 report. Numbers are from single-seed (seed 0) runs unless stated.
Items marked **[TODO]** are pending.

---

## 1. Motivation & research question

Retrieval-augmented generation (RAG) systems trust the passages they retrieve. If a passage has been edited
to give a wrong answer, the LLM usually repeats it. A natural defence is an **evidence classifier** that
labels each (question, passage) pair as **supporting / misleading / irrelevant** before generation.

These classifiers are trained on *synthetic* misleading passages, which are made by swapping the answer
entity or by asking an LLM to rewrite. **Our question:** does the classifier learn that the passage
*gives the wrong answer to this question*, or does it learn the *traces of the editing process*
(an odd entity type, unnatural wording)? If it learns the traces, it will fail on fakes made a different way.

## 2. What is new (innovation)

1. **Bystander test** (main diagnostic). We take the *same* fake passage and pair it with a *different*
   question whose answer was not touched. The correct label is *supporting*. A model that flags it as
   misleading is reacting to the edit, not to the question.
2. **2×2 difference-in-differences (DiD) metric.** It crosses {fake, original passage} with
   {target question, bystander question}:
   DiD = [P(M | q, fake) − P(M | q, orig)] − [P(M | bystander q, fake) − P(M | bystander q, orig)].
   It cancels both the "edit fingerprint" effect and any question-only shortcut, so it measures only the
   genuine question–content signal (0 means none).
3. **Generator-controlled evaluation.** We use three in-house fake generators of increasing realism
   (type-mismatched swap → same-type swap → LLM rewrite) plus two external benchmarks (NQ-Swap,
   FaithEval). This lets us test the model across ways of making fakes.
4. **Placebo test.** The same edit is applied to a *non-answer* entity, so the correct label is *supporting*.
5. **Pair training (our modification).** Half of the "supporting" training examples are bystander
   examples, which forces the model to look at the question.
6. **Attribution check.** Gradient×input share on the edited span under the true vs the bystander question.
7. **Cross-document consistency (improvement).** The classifier also receives a reference snippet from
   *other* paragraphs of the same article, so it can detect *conflict* instead of needing world knowledge
   (Section 7).

## 3. Data

| Item | Source | Size used |
|---|---|---|
| Questions/passages | SQuAD v1.1 (Rajpurkar et al., 2016) | train 1,135 records, dev 234, test 746 |
| Split | train/dev split **by article title** (no article overlap); SQuAD validation = test | |
| Entity types | spaCy `en_core_web_sm` NER → PER, ORG, LOC, DATE, NUM, MISC | |
| External test 1 | NQ-Swap (Longpre et al., 2021), dev | 500 |
| External test 2 | FaithEval-counterfactual (Ming et al., 2024) | 500 |

Each record yields:
- **supporting**: the original passage
- **misleading**: the passage with the answer replaced
- **irrelevant**: a passage from another question. Half are from the same article ("hard"), half from a
  random article ("easy").

**Lean scale:** we kept the shortest passages that had a valid LLM fake, so that each run takes about
25 minutes on a 16 GB Apple M5 laptop. This is a limitation (Section 9).

### Fake generators

| Generator | How | Realism |
|---|---|---|
| `swap_mismatch` | answer → entity of a **different** type (e.g. person → number) | low; easy fingerprint |
| `swap_typed` | answer → entity of the **same** type from the corpus | medium |
| `llm_rewrite` | Qwen2.5-1.5B-Instruct rewrites the passage so it states the same-type fake answer | high |

LLM outputs were validated: the old answer is gone, the bystander answer is kept, and the length ratio
is 0.85–1.3. A second sampling pass raised the valid test fakes from 231 to 746.

## 4. Model & training

- **Backbone:** `cross-encoder/nli-deberta-v3-small` (DeBERTa-v3-small already fine-tuned on NLI).
  Fine-tuned, not trained from scratch, as the handout requires.
  - Plain `microsoft/deberta-v3-small` could not separate supporting from misleading (AUROC ≈ 0.5, loss
    stuck at ≈ 0.46). This is reported in an appendix (`runs_plain_deberta/`).
- **Input:** `[CLS] question [SEP] passage [SEP]` → 3 classes.
- **Hyperparameters:**
  - AdamW, lr 5e-5, weight decay 0.01, linear warm-up of 10%
  - 3 epochs, effective batch 16, max_len 320, fp32
- **Model selection:** best dev macro-F1. The test sets are never used for selection.
- **Memory engineering for 16 GB unified memory (MPS):**
  - frozen word-embedding table (98M of 141M parameters)
  - micro-batching (gradient accumulation of 2)
  - padding to a multiple of 64
  - `foreach=False` AdamW
  - periodic cache clearing and an MPS watermark of 0.8

### Training regimes

| Regime | Supporting class | Purpose |
|---|---|---|
| base | original passages | standard approach |
| pair (ours) | 50% original, 50% bystander | force use of the question |
| placebo | 50% original, 50% placebo edit | teach "edits are not enough", with no question shortcut |
| mix | base, with a random generator per record | does seeing variety help? |
| qblind | base, with the question removed | how much can be done without the question? |

## 5. Metrics

| Metric | Meaning |
|---|---|
| TPR | % of fake passages flagged misleading (detection rate) |
| FAR_orig | % of original passages wrongly flagged |
| FAR_by | % of bystander pairs wrongly flagged (the text is identical to the fake!) |
| FAR_pl | % of placebo edits wrongly flagged |
| AUC_orig | AUROC fake vs original (the "standard" number papers report) |
| **DiD** | **headline**: genuine question–content signal (Section 2) |

## 6. Main results (single-passage classifier)

### 6.1 Baseline looks good on the standard metric but mostly detects edits

| Train → test | TPR | FAR_orig | FAR_by | AUC_orig | DiD |
|---|---|---|---|---|---|
| mismatch base → mismatch | 65.5% | 5.1% | **56.7%** | 0.890 | 0.102 |
| mismatch base → typed | 36.7% | 5.1% | 35.9% | 0.730 | 0.021 |
| mismatch base → llm | 21.0% | 5.1% | 20.6% | 0.632 | 0.017 |
| mismatch base → NQ-Swap | 15.0% | 6.8% | – | 0.553 | – |
| mismatch base → FaithEval | 1.6% | – | – | – | – |

**Takeaway:** an AUROC of 0.89 looks strong. But the model flags 57% of bystander pairs, where the text is
identical to the fake and only the question changed. Most of its "detection" is recognising that the
passage was edited.

### 6.2 Effect of training regime (DiD; test generators mismatch / typed / llm)

| Train gen / regime | DiD mismatch | DiD typed | DiD llm | FAR_orig | Notes |
|---|---|---|---|---|---|
| mismatch base | .102 | .021 | .017 | .051 | |
| **mismatch pair (ours)** | **.200** | **.105** | **.062** | .251 | DiD doubles; FAR_by 56.7% → 22.5% |
| mismatch placebo | .004 | .001 | −.038 | .221 | collapses to chance |
| mismatch qblind | 0 | 0 | 0 | .315 | AUC_orig .569 |
| mix base | .038 | .015 | −.019 | .198 | variety alone does not help |
| mix pair | .076 | .050 | .004 | .334 | |
| typed base | .070 | .031 | .020 | .185 | |
| typed pair | .024 | .015 | −.003 | .666 | |
| llm base | ≈ −.03 | | | .547 | |
| llm pair | ≈ 0 | | | .610 | |

### 6.3 Attribution (edited-span share, true question ÷ bystander question; >1 = uses the question)

| Model | mismatch | typed | llm |
|---|---|---|---|
| mismatch base | 1.06 | 1.19 | 1.24 |
| mismatch pair | 1.58 | 2.03 | 1.89 |
| mix base | 1.29 | 1.31 | 1.40 |
| mix pair | 1.54 | 1.90 | 2.02 |

### 6.4 Interpretation

1. **Standard detectors learn the generator, not the misinformation.** High AUROC is driven by the
   edit fingerprint, and transfer to other generators and to external benchmarks is near chance.
2. **Mixing generators does not fix this.**
3. **Pair training doubles the genuine signal** (DiD 10 → 20 points). Attribution confirms the model
   looks at the edited span more when the question is about it.
   - **Caveat:** part of pair training's gain in plain "CS" came from a *question shortcut* (primary vs
     bystander question style). Our 2×2 DiD was designed to catch exactly this.
4. **Placebo training (shortcut-free) learns nothing.** Once fingerprints and question shortcuts are
   both removed, there is no signal left. Same-type and LLM fakes are undetectable **from the passage
   alone**, because a plausible same-type swap can only be spotted with external knowledge.
5. **Practical verdict:** neither single-passage model is deployable.
   - The baseline is better on in-distribution fakes: it catches 65% vs 48%, with false alarms of
     5% vs 25%.
   - Both fail on external fakes.

## 7. Improvement: cross-document consistency

**Motivation.** Section 6 says the bottleneck is *knowledge*. Following work on conflict detection in RAG
(Hong et al., 2024; MIRAGE; ConflictRAG), we give the classifier **independent evidence**: a reference
snippet from *other paragraphs of the same article*. Detection then becomes "does this passage conflict
with the reference?" rather than "do I know this fact?".

**Method.**
- Input: `[CLS] question [SEP] Reference: <snippet> [SEP] candidate passage [SEP]`, with max_len 448.
- The candidate's own paragraph is never used as the reference.
- Snippet: the top-2 two-sentence windows by IDF-weighted word overlap with the question.
- **Good-retriever condition (main):** if the article contains a window mentioning the true answer, that
  window is included. Coverage: 39% of questions in train and in test.
- **Keyword-retriever condition (realistic test-time check):** plain retrieval only. Coverage: 11%.
- Bystander examples receive the reference retrieved for the bystander question, so the 2×2 DiD still
  applies.
- The same base regime and hyperparameters are used. Training runs on swap_typed, swap_mismatch and
  llm_rewrite.
- Results are reported separately for **corroborated** (the reference contains the true answer) and
  **non-corroborated** questions. The old single-passage models are scored on the same split.

**Hypothesis.** DiD rises sharply on corroborated questions, including same-type and LLM fakes that were
undetectable before. On non-corroborated questions it stays near the single-passage level.

**Results** (single seed; 289 corroborated / 457 non-corroborated test questions; files `results/xdoc_auc.csv`,
`results/xdoc.csv`, `results/calibrated.csv`).

AUROC, fake vs original passage (threshold-free; 0.5 = chance). Same-type-trained model:

| Test fakes | No reference | Good retriever, corroborated | Good retriever, not corroborated | Keyword retriever, corroborated (n=71) |
|---|---|---|---|---|
| same-type swap | 0.573 | **0.704** | 0.567 | 0.694 |
| LLM rewrite | 0.525 | **0.644** | 0.504 | 0.641 |
| mismatch swap | 0.595 | **0.712** | 0.578 | 0.727 |

Other training generators (corroborated group, AUROC fake vs original, tested on same-type / LLM):
- mismatch-trained: 0.725 / 0.622 without a reference, 0.625 / 0.569 with one (worse; on its own mismatch fakes
  it drops 0.872 → 0.691)
- LLM-trained: 0.497 / 0.515 without, 0.620 / 0.593 with

Findings:
1. **Evidence helps exactly where it exists.** On corroborated questions, AUROC rises by about 0.12–0.13 for the
   same-type and LLM fakes that the single-passage model could not detect. On non-corroborated questions the
   gain is ≈0. This supports Section 6's conclusion that the bottleneck is *knowledge*, not model capacity.
   The gain survives realistic keyword retrieval.
2. **It is passage-level, not question-level.** AUROC fake vs bystander stays ≈0.50–0.56: the model flags a
   passage that conflicts with the article whatever the question asks.
3. **Default (argmax) predictions over-flag.** Genuine passages were flagged 33–63% of the time, which made the
   argmax DiD look large (e.g. 0.25).
   - We calibrated the threshold on dev to a 10% false-alarm rate. At that point, detection of same-type and
     LLM fakes is only 7–14% and DiD is ≤ 0.07.
   - So the gain is real in ranking terms but small at strict operating points.
4. **A reference can dilute an easy cue.** For the mismatch-trained model, the longer input weakened the
   type-mismatch fingerprint. LLM-trained models learned little in either setting (dev macro-F1 0.61).

**Conclusion.** Supplying independent evidence turns undetectable same-type and LLM fakes into partially
detectable ones (AUROC ≈0.70), but only when corroborating evidence exists. It is still not reliable at low
false-alarm rates. Next steps: more data, question-aware conflict training (a bystander-with-reference pair
regime), and stronger retrieval.

## 8. Compliance with the project handout

- Uses a pretrained transformer that is fine-tuned, not trained from scratch.
- Uses public datasets (SQuAD, NQ-Swap, FaithEval) with credit.
- Reviews existing techniques (Section 10) and proposes a modification (pair training, cross-document
  input) with an evaluation of it.
- Code is in `src/`, with a one-command pipeline (`run_all.sh`, `run_train.sh`, `run_xdoc.sh`).
- **[TODO]** AI Use Statement, group member names/numbers, and the 10-page limit.

## 9. Limitations

- **Single seed:** the differences of a few points (e.g. DiD 0.02 vs 0.03) are within noise. Extra seeds
  are recommended for the headline comparison.
- **Lean scale:** only the shortest passages, ~1.1k training records. The results may not hold for long
  passages.
- **LLM fakes come from a small 1.5B model;** stronger LLMs may leave different traces.
- **Coarse NER typing:** spaCy types are coarse, so some "same-type" swaps are still odd.
- **Cross-document references:**
  - the "good retriever" is an upper bound
  - realistic keyword retrieval finds corroborating evidence for only 11% of questions
  - same-article references are easier than open-web retrieval
- **External benchmarks:** these use different question styles, so their scores mix domain shift with
  generator shift.

## 10. Related work (verify before citing)

- **Hong et al. (2024).** *Why So Gullible? Enhancing the Robustness of Retrieval-Augmented Models against
  Counterfactual Noise.* Findings of NAACL. Discriminator for misleading passages in RAG.
- **Longpre et al. (2021).** *Entity-Based Knowledge Conflicts in Question Answering.* EMNLP. (NQ-Swap)
- **Ming et al. (2024).** *FaithEval.* (FaithEval counterfactual)
- **Clark, Yatskar & Zettlemoyer (2019); He, Zha & Wang (2019).** Product-of-Experts debiasing.
- **Kaushik, Hovy & Lipton (2020).** Counterfactually-augmented data.
- **Joshi & He (2021).** An investigation of the (in)effectiveness of counterfactually augmented data.
- **Geirhos et al. (2020).** Shortcut learning in deep neural networks.
- **He et al. (2021).** DeBERTaV3. Rajpurkar et al. (2016), SQuAD. Qwen Team (2024), Qwen2.5.
- **Conflict-aware RAG** (2025–2026): MIRAGE, ConflictRAG, CONFACT, TCR. These are recent preprints;
  check the exact titles and authors.

## 11. AI Use Statement (draft)

> We used Claude (Anthropic) as a coding assistant to help implement the data pipeline, training scripts
> and analysis, and to search related literature. All experimental design decisions, results and
> interpretations were reviewed and verified by the group.

## 12. Reproducibility / files

| Path | Content |
|---|---|
| `src/build_data.py` | SQuAD records, entity edits, bystander/placebo construction |
| `src/llm_fakes.py` | Qwen2.5 rewrites |
| `src/make_sets.py` | train/dev/test sets per generator × regime |
| `src/train_eval.py` | fine-tune + predict |
| `src/analyze.py` | metrics → `results/*.csv` |
| `src/attribution.py` | gradient×input span share |
| `src/xdoc_refs.py`, `src/make_sets_xdoc.py`, `src/analyze_xdoc.py` | cross-document experiment |
| `results/` | all metric tables |
