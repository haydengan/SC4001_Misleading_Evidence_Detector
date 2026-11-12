"""Cross-document consistency: attach a retrieved reference snippet to every record.

For a question we retrieve, from the OTHER paragraphs of the same Wikipedia article (never the candidate's own
paragraph), the top-k two-sentence windows by IDF-weighted word overlap with the question. No answer oracle is used,
so whether the reference actually corroborates the true answer is natural variation; we record it ("corr") and
report results separately for corroborated vs non-corroborated questions.
Main condition = controlled "good retriever" (answer-bearing window forced in when the article has one);
q_bm25/by_bm25 = plain keyword retrieval, used only as a realistic test-time check.
Writes data/refs_{split}.json : {record_id: {"q": ref_for_question, "by": ref_for_bystander_q,
                                            "corr": answer in ref_q, "corr_by": bystander answer in ref_by}}
"""
import json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STOP = set("""a an the of in on at to for from by with and or but is are was were be been being what which who whom
whose when where why how did do does that this these those it its as into than then there their they he she his her
him we you your our i not no can could would should may might will shall has have had about after before during
over under between most more many much some any one two first name type kind year""".split())
TOK = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]
occurs = lambda t, s: re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", t, flags=re.I) is not None


def windows(par):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", par.strip()) if s]
    return [" ".join(sents[i:i + 2]) for i in range(max(1, len(sents) - 1))]


def retrieve(q, wins, idf, k, answer=None):
    """Top-k windows by IDF overlap. With `answer` (controlled 'good retriever' condition), the best-scoring window
    that mentions the answer is forced into the top-k if one exists anywhere in the article."""
    if not wins:
        return ""
    qt = set(TOK(q))
    scored = sorted(((sum(idf[w] for w in qt & set(TOK(x))), i) for i, x in enumerate(wins)), reverse=True)
    picked = [i for _, i in scored[:k]]
    if answer is not None and not any(occurs(wins[i], answer) for i in picked):
        hit = next((i for _, i in scored if occurs(wins[i], answer)), None)
        if hit is not None:
            picked = [hit] + picked[:k - 1]
    return " ... ".join(wins[i] for i in sorted(picked))  # keep document order


def main(k=2):
    paras = defaultdict(set)
    for split in ["train", "validation"]:
        for d in load_dataset("rajpurkar/squad")[split]:
            paras[d["title"]].add(d["context"])
    for split in ["train", "dev", "test"]:
        recs = [json.loads(l) for l in open(DATA / f"records_{split}.jsonl")]
        out, cov = {}, Counter()
        for r in recs:
            others = [p for p in paras[r["title"]] if p.strip() != r["context"].strip()]
            wins = [w for p in others for w in windows(p)]
            df = Counter(w for x in wins for w in set(TOK(x)))
            idf = defaultdict(lambda: math.log(len(wins) + 1), {w: math.log((len(wins) + 1) / (c + 0.5)) for w, c in df.items()})
            rq, rb = retrieve(r["question"], wins, idf, k, r["answer"]), retrieve(r["bystander_q"], wins, idf, k, r["bystander_a"])
            kq, kb = retrieve(r["question"], wins, idf, k), retrieve(r["bystander_q"], wins, idf, k)  # realistic
            out[r["id"]] = {"q": rq, "by": rb, "corr": occurs(rq, r["answer"]), "corr_by": occurs(rb, r["bystander_a"]),
                            "q_bm25": kq, "by_bm25": kb, "corr_bm25": occurs(kq, r["answer"])}
            cov["corr"] += out[r["id"]]["corr"]; cov["corr_by"] += out[r["id"]]["corr_by"]; cov["bm25"] += out[r["id"]]["corr_bm25"]
            cov["words"] += len(rq.split())
        json.dump(out, open(DATA / f"refs_{split}.json", "w"))
        n = len(recs)
        print(f"[{split}] n={n} corroborated {cov['corr']/n:.2f}  bystander {cov['corr_by']/n:.2f}  keyword-only {cov['bm25']/n:.2f}  avg ref words {cov['words']/n:.0f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
