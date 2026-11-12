"""Cross-document results: detection (TPR), false alarms and DiD, split by whether the retrieved reference
corroborates the true answer. Old single-passage runs are scored on the same record split for a fair comparison.
Writes results/xdoc.csv
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENS = ["swap_mismatch", "swap_typed", "llm_rewrite"]
REFS = json.load(open(ROOT / "data" / "refs_test.json"))


def metrics(path, prefix=""):
    by = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        rid = r["uid"].split("|")[0]
        if rid not in REFS or not r["kind"].startswith(prefix) or (not prefix and r["kind"].startswith("bm25:")):
            continue
        grp = "corr" if REFS[rid]["corr_bm25" if prefix else "corr"] else "nocorr"
        by[(grp, r["gen"], r["kind"][len(prefix):])].append(int(np.argmax(r["p"])) == 1)
    out = []
    for grp in ["corr", "nocorr"]:
        fo, fbo = np.mean(by[(grp, "-", "orig")]), np.mean(by[(grp, "-", "bystander_orig")])
        for g in GENS:
            tpr, fby = np.mean(by[(grp, g, "fake")]), np.mean(by[(grp, g, "bystander")])
            out.append({"group": grp, "test_gen": g, "n": len(by[(grp, g, "fake")]), "TPR": tpr, "FAR_by": fby,
                        "FAR_orig": fo, "DiD": (tpr - fo) - (fby - fbo)})
    return out


def main():
    rows = []
    runs = [("none (single passage)", ROOT / "runs" / f"{g}_base_s0" / "preds.jsonl", "", g) for g in GENS]
    runs += [("good retriever", ROOT / "runs" / f"xdoc_{g}_base_s0" / "preds.jsonl", "", g) for g in GENS]
    runs += [("keyword retriever", ROOT / "runs" / f"xdoc_{g}_base_s0" / "preds.jsonl", "bm25:", g) for g in GENS]
    for ref, p, prefix, tg in runs:
        if p.exists():
            rows += [{"reference": ref, "train_gen": tg, **m} for m in metrics(p, prefix)]
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "xdoc.csv", index=False)
    pd.set_option("display.width", 200)
    piv = df.pivot_table(index=["reference", "train_gen", "group"], columns="test_gen", values=["TPR", "DiD"]).round(3)
    print(piv)


if __name__ == "__main__":
    main()
