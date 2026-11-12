# SC4001 Group Project — Project Plan

**Track:** F. Reliable Retrieval Augmented Generation
**Deadline:** 16 Nov 2026, 11:59 PM (PDF report ≤10 pages + code ZIP on NTULearn)
**Plan written:** 17 Oct 2026

---

## 1. Topic

**Knowing when not to answer: a trained evidence filter for RAG under noisy and misleading documents**

### Research question
> When retrieved documents are irrelevant, missing, or misleading, can a small trained evidence scorer (filtering documents + abstaining) reduce wrong answers from a frozen LLM, compared with prompt-only abstention?

### Why this problem
- A RAG system retrieves documents and feeds them to an LLM. If the documents are bad, the LLM usually answers confidently anyway.
- GRAB-RAG (arXiv 2608.22228, Aug 2026) tested small frozen models (3.8–8B) on NQ and HotpotQA:
  - Models abstain fine when evidence is **missing**.
  - But they still answer **41.6% of misleading questions** even when told to abstain (63% of those repeat the planted wrong answer).
  - Chain-of-thought barely helps; their verifiers either throw away too many correct answers or get fooled.
  - Conclusion: prompting checks whether evidence is *sufficient*, not whether it is *correct*. → **Open gap we target.**
- Industry context: abstention/calibration is a known weak point across pipeline, agentic, and GraphRAG systems.

---

## 2. Pipeline

```
Question + documents ──► [Evidence scorer (trained, ours)] ──► keep high-scoring docs ──► Frozen LLM ──► Answer
                                      │
                                      └─► if no doc scores above threshold ──► "I don't know"
```

- **LLM:** off-the-shelf, frozen, not trained (e.g. Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct in 4-bit).
- **Evidence scorer (the neural network we train):** small cross-encoder (e.g. MiniLM / DeBERTa-small), input = (question, document), output = "does this document support answering the question?"
- **Supervised training labels:** HotpotQA marks which paragraphs are supporting → free labels.

---

## 3. Data

**HotpotQA, distractor setting.** Each question comes with 10 paragraphs: 2 supporting + 8 distractors. No search index needed.

| Split | Use |
|---|---|
| HotpotQA train | Train the evidence scorer |
| Held-out part of train | Dev set: tune threshold / hyperparameters |
| HotpotQA validation (~1,000 sampled) | **Our test set**, never used for tuning |

### Test conditions (same questions under each, for paired comparison)

| Condition | What the LLM gets | Correct behaviour |
|---|---|---|
| **Gold** | Only the 2 supporting paragraphs | Answer correctly |
| **Noisy** | All 10 paragraphs (2 good + 8 irrelevant) | Answer correctly, ignore noise |
| **Missing** | Only the 8 distractors | Say "I don't know" |
| **Misleading** | Supporting paragraphs with the true answer swapped for a fake one | Not be fooled (abstain) |

Misleading example: "*1984* is a novel by **George Orwell**" → "*1984* is a novel by **Aldous Huxley**".

Filter for usable questions: skip yes/no answers; answer must appear in a supporting paragraph and not in any distractor (so "Missing" really has no answer).

---

## 4. Systems to compare

1. **No filter:** all docs → LLM, plain prompt
2. **Prompt abstention:** same, but told to say "unanswerable" if unsupported (≈ GRAB-RAG baseline)
3. **Off-the-shelf reranker:** pretrained cross-encoder, no training, + threshold
4. **LLM self-judge:** LLM rates each doc's relevance via prompting
5. **Ours:** trained evidence scorer + threshold abstention

### Possible novel twist for the Misleading case (pick one, or treat Misleading purely as a stress test)
A small model cannot truly *verify facts*. It can only use clues:
- **Cross-document conflict:** do documents disagree on the answer?
- **Conflict with LLM memory:** compare the LLM's no-document answer vs its with-document answer.
- **Train on synthetic fakes:** add swapped docs as negatives. ⚠️ Risk: learns artefacts of how fakes were made. Test on fakes generated a *different* way.

---

## 5. Metrics

- **Exact match / F1:** answer accuracy
- **Abstention rate:** how often it says "I don't know" (want high on Missing/Misleading, low on Gold/Noisy)
- **Fooled rate:** % of Misleading questions answered with the planted fake answer
- **Accuracy vs coverage curve:** sweep the abstention threshold
- **Cost:** tokens sent to the LLM (filtering should reduce this)

---

## 6. Experiments / ablations

- Number of distractors: 0 / 4 / 8
- Abstention threshold sweep
- LLM size: 1.5B vs 7B (does the filter help small models more?)
- Scorer trained with vs without synthetic fake negatives
- Failure analysis: (a) filter drops a genuinely useful doc; (b) LLM ignores good evidence; (c) fakes that still get through, and why

Negative results are acceptable if well supported (handout says so).

---

## 7. Timeline

| Dates | Work |
|---|---|
| 17–23 Oct | Data prep + 4 conditions; run baselines 1–2 |
| 24–30 Oct | Train evidence scorer; baselines 3–4 |
| 31 Oct–6 Nov | Main experiments + ablations |
| 7–13 Nov | Failure analysis; write report |
| 14–16 Nov | Buffer, final checks, submit |

⚠️ If work only resumes ~31 Oct, weeks 1–2 must be compressed. Prioritise: baseline + conditions → scorer → main comparison. Drop ablations first if short on time.

---

## 8. Compute

- Colab / course GPU (T4 or L4, 16 GB). LLM is inference-only; scorer training < 1 hr.
- 7B LLM needs 4-bit loading on a T4.
- Save checkpoints and results regularly (Colab disconnects).

---

## 9. Report checklist (from handout)

- [ ] Cover: team names + group number
- [ ] Introduction
- [ ] Review of existing techniques (RAG, RGB benchmark, Self-RAG/CRAG, GRAB-RAG, rerankers)
- [ ] Methods
- [ ] Experiments & results
- [ ] Discussion + limitations + failure cases
- [ ] AI Use Statement (tools, how outputs were verified, team's contributions)
- [ ] Code: commented, README with dependencies, data prep, commands
- [ ] A4, single column, Arial 10pt, ≤10 pages (excl. cover, contents, references, appendix)
- [ ] Credit all existing code / pretrained models used

Grading: Experiments 40% · Presentation 30% · Novelty 20% · Peer review 10%

---

## 10. References to check (verify before citing)

- GRAB-RAG: "Prompt-Based Abstention Fails Under Misleading Context" — https://arxiv.org/abs/2608.22228 (code "to be released"; single author, recent preprint)
- RGB benchmark: noise robustness, negative rejection, counterfactual robustness — https://arxiv.org/abs/2309.01431
- LIT-RAGBench (LREC 2026), includes abstention category — https://arxiv.org/abs/2603.06198
- HotpotQA (Yang et al., 2018), distractor setting
- Self-RAG, CRAG (Corrective RAG): related methods for the literature review
- Industry context: PageIndex ("vectorless" RAG), agentic RAG, GraphRAG

---

## 11. Current state of the folder

- `project_handout.pdf`: assignment handout
- `PROJECT_PLAN.md`: this file
- `.venv/`: Python 3.11 env with torch, transformers, accelerate, datasets (for local testing on Mac; main runs on Colab)
- `src/`, `data/`, `results/`: empty, ready for code

**Next step when resuming:** write the data prep script (HotpotQA → 4 conditions) and run the no-filter baseline.
