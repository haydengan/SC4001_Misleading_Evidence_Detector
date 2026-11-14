"""Cross-document sets: same examples as make_sets.py, plus a "ref" snippet from other paragraphs of the article.

Reference for a (question, passage) example is retrieved for THAT question (bystander examples get the bystander
question's reference), so the model can check the candidate against independent evidence.
Writes data/sets_xdoc/{train,dev}_{gen}_{base,pair}.jsonl, test_squad.jsonl (good-retriever refs),
test_squad_bm25.jsonl (keyword-retrieval refs), test_external.jsonl (no reference available).
"""
import json, random, zlib
from pathlib import Path
import make_sets as M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sets_xdoc"
BY_Q = {"bystander", "bystander_orig"}


def attach(rows, refs, mode=""):
    for x in rows:
        r = refs.get(x["rid"])
        if r is None:
            x["ref"], x["corr"] = "", None
            continue
        x["ref"] = r[("by" if x["kind"] in BY_Q else "q") + mode]
        x["corr"] = bool(r["corr" + mode])  # does the question's reference contain the true answer
        x["refmode"] = mode or "good"
    return rows


def dump(rows, name):
    with open(OUT / name, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = {s: json.load(open(ROOT / "data" / f"refs_{s}.json")) for s in ["train", "dev", "test"]}
    tr, dv, te = M.load_split("train"), M.load_split("dev"), M.load_split("test")
    for gen in M.GENS + ["mix"]:
        # pair = 50% of supporting examples are bystanders (fake passage + untouched question, with that
        # question's own reference): teaches "conflicts with the article, but not about MY question -> supporting"
        for regime in ["base", "pair"]:
            dump(attach(M.train_examples(tr, gen, regime, random.Random(zlib.crc32(f"{gen}{regime}".encode()))),
                        refs["train"]), f"train_{gen}_{regime}.jsonl")
            dump(attach(M.train_examples(dv, gen, regime, random.Random(1)), refs["dev"]), f"dev_{gen}_{regime}.jsonl")
    dump(attach(M.test_examples(te), refs["test"]), "test_squad.jsonl")
    bm = attach(M.test_examples(te), refs["test"], "_bm25")
    for x in bm:
        x["kind"] = "bm25:" + x["kind"]
    dump(bm, "test_squad_bm25.jsonl")
    ext = [json.loads(l) for l in open(ROOT / "data" / "sets" / "test_external.jsonl")]
    dump(attach(ext, {}), "test_external.jsonl")
    print("corroborated share: train %.2f test %.2f" % (sum(refs["train"][r["id"]]["corr"] for r in tr) / len(tr),
                                                        sum(refs["test"][r["id"]]["corr"] for r in te) / len(te)))


if __name__ == "__main__":
    main()
