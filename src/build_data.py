"""Stage 1: build typed SQuAD records with rule-based fakes, placebos and bystanders.

For every selected question we store:
  - the original passage (supporting)
  - for each rule generator: the fake passage (misleading) and a placebo passage (supporting)
  - a bystander question on the same paragraph (used to test fake passages under another question)
  - irrelevant passages (easy: other article, hard: other paragraph of the same article)
  - the edit plan (old -> new) that the LLM generator executes in stage 2 (llm_fakes.py)

Splits: SQuAD-train is split BY ARTICLE into train/dev; SQuAD-validation (unseen articles) is the test set.
Output: data/records_{train,dev,test}.jsonl
"""
import json, random, re, argparse
from collections import defaultdict
from pathlib import Path
import spacy
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COARSE = {
    "PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
    "DATE": "DATE", "TIME": "DATE",
    "CARDINAL": "NUM", "QUANTITY": "NUM", "MONEY": "NUM", "PERCENT": "NUM", "ORDINAL": "NUM",
    "NORP": "MISC", "EVENT": "MISC", "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC", "PRODUCT": "MISC",
}


def occurs(text, s):
    """Word-boundary, case-insensitive occurrence test."""
    return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, flags=re.I) is not None


def replace_all(text, old, new):
    return re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", lambda m: new, text, flags=re.I)


def overlap(a, b):
    a, b = a.lower(), b.lower()
    return a in b or b in a or len(set(a.split()) & set(b.split())) > 0


def answer_type(a, emap):
    """Coarse type of answer a: exact entity, entity after stripping an article, or entity covering most of a."""
    if a in emap:
        return emap[a]
    a2 = re.sub(r"^(the|a|an)\s+", "", a, flags=re.I).strip(" .,")
    if a2 in emap:
        return emap[a2]
    for t, ty in emap.items():
        if t in a and len(t) >= 0.6 * len(a):
            return ty
    if re.fullmatch(r"[\d.,]+( (million|billion|thousand|percent|%))?", a2):
        return "NUM"
    return None


class Builder:
    def __init__(self, rng, type_pool, by_title):
        self.rng, self.type_pool, self.by_title = rng, type_pool, by_title
        self.types = list(type_pool)

    def pick(self, ty, ctx, avoid, mismatch=False):
        """Replacement entity of the same (or, if mismatch, a different) type that is not already in ctx."""
        types = [t for t in self.types if t != ty] if mismatch else [ty]
        for _ in range(50):
            cand = self.rng.choice(self.type_pool[self.rng.choice(types)])
            if not occurs(ctx, cand) and not any(overlap(cand, x) for x in avoid):
                return cand
        return None

    def make(self, q, c, qs, emap):
        rng = self.rng
        a = q["answers"]["text"][0].strip()
        ty = answer_type(a, emap)
        if ty is None or ty not in self.type_pool or not occurs(c, a):
            return None
        # bystander: another question on the same paragraph whose answer survives editing `a`
        bys = [o for o in qs if o["id"] != q["id"]
               and not overlap(o["answers"]["text"][0], a) and occurs(c, o["answers"]["text"][0].strip())]
        if not bys:
            return None
        by = rng.choice(bys)
        a2 = by["answers"]["text"][0].strip()
        # placebo target: another typed entity, not in the question, not overlapping either answer
        cands = [(t, tt) for t, tt in emap.items()
                 if not overlap(t, a) and not overlap(t, a2) and not occurs(q["question"], t) and len(t) > 2]
        if not cands:
            return None
        p_old, p_ty = rng.choice(cands)
        avoid = [a, a2, p_old]
        new_typed, new_mis = self.pick(ty, c, avoid), self.pick(ty, c, avoid, mismatch=True)
        p_typed, p_mis = self.pick(p_ty, c, avoid), self.pick(p_ty, c, avoid, mismatch=True)
        if None in (new_typed, new_mis, p_typed, p_mis):
            return None
        other_titles = [t for t in self.by_title if t != q["title"]]
        irr_easy = rng.choice(self.by_title[rng.choice(other_titles)])
        same = [p for p in self.by_title[q["title"]] if p != c and not occurs(p, a)]
        rec = {
            "id": q["id"], "title": q["title"], "question": q["question"], "answer": a, "type": ty,
            "context": c, "bystander_q": by["question"], "bystander_a": a2,
            "irr_easy": irr_easy, "irr_hard": rng.choice(same) if same else irr_easy,
            "irr_hard_is_same_article": bool(same),
            "edits": {"fake": {"old": a, "typed": new_typed, "mismatch": new_mis},
                      "placebo": {"old": p_old, "type": p_ty, "typed": p_typed, "mismatch": p_mis}},
            "passages": {
                "swap_typed": {"fake": replace_all(c, a, new_typed), "placebo": replace_all(c, p_old, p_typed)},
                "swap_mismatch": {"fake": replace_all(c, a, new_mis), "placebo": replace_all(c, p_old, p_mis)},
            },
        }
        # sanity: fake removes the answer, placebo keeps it, bystander answer survives the fake edit
        for g in rec["passages"].values():
            if occurs(g["fake"], a) or not occurs(g["placebo"], a) or not occurs(g["fake"], a2):
                return None
        return rec


def main(n_train, n_dev, n_test, seed):
    rng = random.Random(seed)
    sq = load_dataset("rajpurkar/squad")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])

    titles = sorted(set(sq["train"]["title"]))
    rng.shuffle(titles)
    dev_titles = set(titles[: len(titles) // 10])
    pools = {"train": [], "dev": [], "test": list(sq["validation"])}
    for ex in sq["train"]:
        pools["dev" if ex["title"] in dev_titles else "train"].append(ex)

    for split, n_q in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        by_ctx = defaultdict(list)
        for ex in pools[split]:
            by_ctx[ex["context"]].append(ex)
        by_title = defaultdict(list)
        for c, qs in by_ctx.items():
            by_title[qs[0]["title"]].append(c)
        ctxs = [c for c, qs in by_ctx.items() if len(qs) >= 2]  # need >=2 questions for a bystander
        rng.shuffle(ctxs)
        ctxs = ctxs[: int(n_q * 1.6)]
        print(f"[{split}] NER on {len(ctxs)} paragraphs ...", flush=True)
        ents = {c: [(e.text, COARSE[e.label_]) for e in doc.ents if e.label_ in COARSE]
                for c, doc in zip(ctxs, nlp.pipe(ctxs, batch_size=64))}
        type_pool = defaultdict(set)  # replacement pools come from the same split only
        for c in ctxs:
            for t, ty in ents[c]:
                if 1 <= len(t.split()) <= 5 and len(t) > 1:
                    type_pool[ty].add(t)
        b = Builder(rng, {k: sorted(v) for k, v in type_pool.items()}, by_title)

        records = []
        for c in ctxs:  # one primary question per paragraph
            if len(records) >= n_q:
                break
            emap = {}
            for t, ty in ents[c]:
                emap.setdefault(t, ty)
            for q in rng.sample(by_ctx[c], len(by_ctx[c])):
                rec = b.make(q, c, by_ctx[c], emap)
                if rec:
                    records.append(rec)
                    break
        with open(DATA / f"records_{split}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        tc = defaultdict(int)
        for r in records:
            tc[r["type"]] += 1
        print(f"[{split}] {len(records)} records  types={dict(tc)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_dev", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    main(a.n_train, a.n_dev, a.n_test, a.seed)
