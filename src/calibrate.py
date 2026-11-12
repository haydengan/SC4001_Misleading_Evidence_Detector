"""Threshold calibration: flag 'misleading' only if p(M) >= t, with t chosen on the DEV split so that 10% of
original (genuine) dev passages are flagged. Applied once to test; no retraining, no test-set tuning.
Writes results/calibrated.csv (same layout as xdoc.csv).
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_eval as T
from analyze_xdoc import REFS, GENS

TARGET = 0.10


def dev_threshold(run, sets):
    tok = AutoTokenizer.from_pretrained(ROOT / "runs" / run / "model")
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / "runs" / run / "model").to(T.DEV)
    gen = run.replace("xdoc_", "").rsplit("_base", 1)[0]
    dev = [json.loads(l) for l in open(ROOT / sets / f"dev_{gen}_base.jsonl")]
    dev = [d for d in dev if d["kind"] == "orig"]
    p = T.predict(model, dev, T.make_collate(tok, False, 448 if "xdoc" in run else 320))[:, 1]
    del model; torch.mps.empty_cache() if T.DEV == "mps" else None
    return float(np.quantile(p, 1 - TARGET))


def metrics(path, t, prefix=""):
    by = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        rid = r["uid"].split("|")[0]
        if rid not in REFS or not r["kind"].startswith(prefix) or (not prefix and r["kind"].startswith("bm25:")):
            continue
        grp = "corr" if REFS[rid]["corr_bm25" if prefix else "corr"] else "nocorr"
        by[(grp, r["gen"], r["kind"][len(prefix):])].append(r["p"][1] >= t)
    out = []
    for grp in ["corr", "nocorr"]:
        fo, fbo = np.mean(by[(grp, "-", "orig")]), np.mean(by[(grp, "-", "bystander_orig")])
        for g in GENS:
            tpr, fby = np.mean(by[(grp, g, "fake")]), np.mean(by[(grp, g, "bystander")])
            out.append({"group": grp, "test_gen": g, "TPR": tpr, "FAR_by": fby, "FAR_orig": fo,
                        "DiD": (tpr - fo) - (fby - fbo)})
    return out


def main():
    rows = []
    for g in GENS:
        for run, sets, refs in [(f"{g}_base_s0", "data/sets", [("none (single passage)", "")]),
                                (f"xdoc_{g}_base_s0", "data/sets_xdoc",
                                 [("good retriever", ""), ("keyword retriever", "bm25:")])]:
            t = dev_threshold(run, sets)
            print(run, "threshold", round(t, 3), flush=True)
            for ref, prefix in refs:
                rows += [{"reference": ref, "train_gen": g, "threshold": t, **m}
                         for m in metrics(ROOT / "runs" / run / "preds.jsonl", t, prefix)]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "calibrated.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["reference", "train_gen", "group"], columns="test_gen",
                         values=["TPR", "DiD"]).round(3).to_string())
    print(df.pivot_table(index=["reference", "train_gen", "group"], values=["FAR_orig"]).round(3).to_string())


if __name__ == "__main__":
    main()
