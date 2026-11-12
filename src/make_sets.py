"""Stage 3: assemble train/dev/test jsonl files.

Each example: {"uid","q","p","label","gen","kind","rid"}; labels 0=supporting 1=misleading 2=irrelevant.
kinds: orig, fake, placebo, bystander (fake passage + other question -> supporting), bystander_orig,
       irr_easy, irr_hard, ext_orig, ext_fake, ext_irr

Training regimes (same size per regime; only the composition of the SUPPORTING class changes):
  base     S = original passages
  pair     S = 50% original, 50% bystander  (same fake passage, other question)  <- our modification
  placebo  S = 50% original, 50% placebo    (same generator applied to a non-answer entity)
  (qblind is the base regime with the question removed; handled in train_eval.py)
"""
import json, random, re, zlib
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SETS = DATA / "sets"
GENS = ["swap_mismatch", "swap_typed", "llm_rewrite"]
S, M, I = 0, 1, 2


def occurs(text, s):
    return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, flags=re.I) is not None


def llm_ok(d, rec):
    """Relaxed validity: edited span gone, answer removed (fake) / kept (placebo), text changed, length close."""
    t, c = d["text"], rec["context"]
    if not t or t.strip() == c.strip() or not 0.85 <= len(t) / len(c) <= 1.3 or occurs(t, d["old"]):
        return False
    if d["kind"] == "fake":
        return not occurs(t, rec["answer"]) and occurs(t, rec["bystander_a"])
    return occurs(t, rec["answer"])


def load_split(split, need_placebo=None):
    """Records with a valid llm fake (and, for the test split, a valid llm placebo)."""
    need_placebo = bool(need_placebo)  # llm placebo is optional; evaluated wherever it is valid
    recs = [json.loads(l) for l in open(DATA / f"records_{split}.jsonl")]
    llm = {}
    p = DATA / f"llm_{split}.jsonl"
    if p.exists():
        recmap = {r["id"]: r for r in recs}
        for d in map(json.loads, open(p)):  # keep the first VALID attempt (attempt 2 = retry), else the first one
            key = (d["id"], d["kind"])
            if key not in llm or (not llm_ok(llm[key], recmap[d["id"]]) and llm_ok(d, recmap[d["id"]])):
                llm[key] = d
    keep = []
    for r in recs:
        f, pl = llm.get((r["id"], "fake")), llm.get((r["id"], "placebo"))
        if f and llm_ok(f, r) and (not need_placebo or (pl and llm_ok(pl, r))):
            r["passages"]["llm_rewrite"] = {"fake": f["text"], "placebo": pl["text"] if pl and llm_ok(pl, r) else None}
            r["llm_exact_new"] = occurs(f["text"], f["new"])
            keep.append(r)
    print(f"[{split}] {len(keep)}/{len(recs)} records kept (valid llm fake{'+placebo' if need_placebo else ''})")
    return keep


def ex(r, q, p, label, gen, kind):
    return {"uid": f"{r['id']}|{gen}|{kind}", "rid": r["id"], "q": q, "p": p, "label": label, "gen": gen,
            "kind": kind, "type": r["type"]}


def train_examples(recs, gen, regime, rng):
    out = []
    for r in recs:
        g = rng.choice(GENS) if gen == "mix" else gen
        fake = r["passages"][g]["fake"]
        sup = ex(r, r["question"], r["context"], S, g, "orig")
        if regime == "pair" and rng.random() < 0.5:
            sup = ex(r, r["bystander_q"], fake, S, g, "bystander")
        if regime == "placebo" and rng.random() < 0.5:
            sup = ex(r, r["question"], r["passages"][g]["placebo"], S, g, "placebo")
        irr_kind = "irr_hard" if rng.random() < 0.5 else "irr_easy"
        out += [sup, ex(r, r["question"], fake, M, g, "fake"), ex(r, r["question"], r[irr_kind], I, g, irr_kind)]
    rng.shuffle(out)
    return out


def test_examples(recs):
    out = []
    for r in recs:
        out += [ex(r, r["question"], r["context"], S, "-", "orig"),
                ex(r, r["bystander_q"], r["context"], S, "-", "bystander_orig"),
                ex(r, r["question"], r["irr_easy"], I, "-", "irr_easy"),
                ex(r, r["question"], r["irr_hard"], I, "-", "irr_hard")]
        for g in GENS:
            ps = r["passages"][g]
            out += [ex(r, r["question"], ps["fake"], M, g, "fake"),
                    ex(r, r["bystander_q"], ps["fake"], S, g, "bystander")]
            if ps["placebo"] is not None:
                out.append(ex(r, r["question"], ps["placebo"], S, g, "placebo"))
    return out


def external(rng, n=500):
    out = []
    nq = list(load_dataset("pminervini/NQ-Swap")["dev"])
    rng.shuffle(nq)
    nq = nq[:n]
    clean = lambda t: re.sub(r"</?[A-Za-z]+>", " ", t).replace("  ", " ").strip()
    for i, d in enumerate(nq):
        r = {"id": f"nq{i}", "type": "-"}
        other = clean(nq[(i + 1) % len(nq)]["org_context"])
        out += [ex(r, d["question"], clean(d["org_context"]), S, "nqswap", "ext_orig"),
                ex(r, d["question"], clean(d["sub_context"]), M, "nqswap", "ext_fake"),
                ex(r, d["question"], other, I, "nqswap", "ext_irr")]
    fe = list(load_dataset("Salesforce/FaithEval-counterfactual-v1.0")["test"])
    rng.shuffle(fe)
    fe = fe[:n]
    for i, d in enumerate(fe):
        r = {"id": f"fe{i}", "type": "-"}
        out += [ex(r, d["question"], d["context"], M, "faitheval", "ext_fake"),
                ex(r, d["question"], fe[(i + 1) % len(fe)]["context"], I, "faitheval", "ext_irr")]
    return out


def dump(rows, name):
    with open(SETS / name, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")


def main():
    SETS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    tr, dv, te = load_split("train"), load_split("dev"), load_split("test")
    n = len(tr)
    for gen in GENS + ["mix"]:
        # placebo-regime ablation only for rule generators (no llm placebos generated for train)
        for regime in ["base", "pair"] + (["placebo"] if gen.startswith("swap") else []):
            dump(train_examples(tr, gen, regime, random.Random(zlib.crc32(f"{gen}{regime}".encode()))),
                 f"train_{gen}_{regime}.jsonl")
            dump(train_examples(dv, gen, regime, random.Random(1)), f"dev_{gen}_{regime}.jsonl")
    dump(test_examples(te), "test_squad.jsonl")
    dump(external(rng), "test_external.jsonl")
    print("train records", n, "test records", len(te))


if __name__ == "__main__":
    main()
