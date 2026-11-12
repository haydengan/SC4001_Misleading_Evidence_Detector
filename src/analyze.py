"""Aggregate runs/*/preds.jsonl into metric tables (results/*.csv) and figures (results/fig_*.png).

Key metrics (misleading score = p[misleading]; prediction = argmax):
  TPR        P(pred=M | fake passage)                         - detection rate
  FAR_by     P(pred=M | bystander: same fake passage, other q) - fingerprint false alarm (text identical!)
  FAR_pl     P(pred=M | placebo: same generator, non-answer edit)
  FAR_orig   P(pred=M | original passage)
  CS         TPR - FAR_by   content sensitivity (0 => pure fingerprint detector)
  AUC_by     AUROC fake vs bystander,  AUC_orig AUROC fake vs original
  DiD        2x2 difference-in-differences (HEADLINE metric):
             [P(M|q,fake) - P(M|q,orig)] - [P(M|bystander q,fake) - P(M|bystander q,orig)]
             removes both the passage-fingerprint effect and any question-only shortcut
             (CS alone is inflated if a model learns "bystander questions -> supporting").
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RUNS, RES = ROOT / "runs", ROOT / "results"
GENS = ["swap_mismatch", "swap_typed", "llm_rewrite"]
SHORT = {"swap_mismatch": "mismatch", "swap_typed": "typed", "llm_rewrite": "llm", "mix": "mix",
         "nqswap": "NQ-Swap", "faitheval": "FaithEval"}


def run_metrics(path):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[(r["gen"], r["kind"])].append(r)
    pm = lambda key: np.array([r["p"][1] for r in by[key]])
    am = lambda key: np.array([int(np.argmax(r["p"])) for r in by[key]])
    out = {}
    orig, borig = am(("-", "orig")), am(("-", "bystander_orig"))
    out["FAR_orig"] = float((orig == 1).mean())
    out["FAR_bystander_orig"] = float((borig == 1).mean())
    irr = np.concatenate([am(("-", "irr_easy")), am(("-", "irr_hard"))])
    out["irr_acc"] = float((irr == 2).mean())
    out["irr_hard_acc"] = float((am(("-", "irr_hard")) == 2).mean())
    for g in GENS:
        tpr = float((am((g, "fake")) == 1).mean())
        far_by = float((am((g, "bystander")) == 1).mean())
        out[f"{g}|TPR"] = tpr
        out[f"{g}|FAR_by"] = far_by
        out[f"{g}|FAR_pl"] = float((am((g, "placebo")) == 1).mean())
        out[f"{g}|CS"] = tpr - far_by
        out[f"{g}|DiD"] = (tpr - out["FAR_orig"]) - (far_by - out["FAR_bystander_orig"])
        f, b, o = pm((g, "fake")), pm((g, "bystander")), pm(("-", "orig"))
        out[f"{g}|AUC_by"] = float(roc_auc_score(np.r_[np.ones(len(f)), np.zeros(len(b))], np.r_[f, b]))
        out[f"{g}|AUC_orig"] = float(roc_auc_score(np.r_[np.ones(len(f)), np.zeros(len(o))], np.r_[f, o]))
        # 3-class macro-F1 on the clean in-distribution mix: orig / fake_g / irrelevant
        ys = [0] * len(orig) + [1] * len(by[(g, "fake")]) + [2] * len(irr)
        ps = np.r_[orig, am((g, "fake")), irr]
        out[f"{g}|macroF1"] = float(f1_score(ys, ps, average="macro"))
    for ext in ["nqswap", "faitheval"]:
        out[f"{ext}|TPR"] = float((am((ext, "ext_fake")) == 1).mean())
        out[f"{ext}|irr_acc"] = float((am((ext, "ext_irr")) == 2).mean())
    o, f = pm(("nqswap", "ext_orig")), pm(("nqswap", "ext_fake"))
    out["nqswap|FAR_orig"] = float((am(("nqswap", "ext_orig")) == 1).mean())
    out["nqswap|AUC_orig"] = float(roc_auc_score(np.r_[np.ones(len(f)), np.zeros(len(o))], np.r_[f, o]))
    out["nqswap|CS"] = out["nqswap|TPR"] - out["nqswap|FAR_orig"]
    return out


def collect():
    recs = []
    for d in sorted(RUNS.iterdir()):
        p = d / "preds.jsonl"
        if not p.exists():
            continue
        m = re.match(r"(.+)_(base|pair|placebo|qblind)_s(\d+)$", d.name)
        gen, regime, seed = m.group(1), m.group(2), int(m.group(3))
        recs.append({"train_gen": gen, "regime": regime, "seed": seed, **run_metrics(p)})
    return pd.DataFrame(recs)


def main():
    RES.mkdir(exist_ok=True)
    df = collect()
    df.to_csv(RES / "all_runs.csv", index=False)
    agg = df.drop(columns="seed").groupby(["train_gen", "regime"]).agg(["mean", "std", "count"])
    agg.to_csv(RES / "agg_mean_std.csv")
    # compact console view
    cols = [f"{g}|{k}" for k in ["TPR", "FAR_by", "DiD"] for g in GENS] + ["nqswap|TPR", "nqswap|CS", "faitheval|TPR",
                                                                           "FAR_orig", "irr_acc"]
    mean = df.drop(columns="seed").groupby(["train_gen", "regime"]).mean()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(mean[cols].round(3).rename(columns=lambda c: c.replace("swap_", "").replace("llm_rewrite", "llm")))
    mean.to_csv(RES / "mean.csv")


if __name__ == "__main__":
    main()
