"""Print compact, screenshot-friendly views of the saved results (no recomputation).
  python src/show_results.py training | main | xdoc | example
"""
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 120)


def training():
    """Per-epoch dev macro-F1 and run time for every run (saved by train_eval.py in runs/<run>/metrics_dev.json)."""
    rows = []
    for d in sorted((ROOT / "runs").iterdir()):
        f = d / "metrics_dev.json"
        if f.exists():
            m = json.load(open(f))
            h = [round(e["dev_macro_f1"], 3) for e in m["history"]]
            rows.append({"run": d.name, "ep0": h[0], "ep1": h[1], "ep2": h[2], "best": round(m["best_dev_macro_f1"], 3),
                         "minutes": round(m["train_time_s"] / 60)})
    print("Dev macro-F1 per epoch (best epoch kept) and training time\n")
    print(pd.DataFrame(rows).to_string(index=False))


def main_table():
    m = pd.read_csv(ROOT / "results" / "mean.csv")
    m = m[~m.train_gen.str.startswith("xdoc")]
    t = pd.DataFrame({"train": m.train_gen.str.replace("swap_", "").str.replace("llm_rewrite", "llm"), "regime": m.regime,
                      "TPR": m["swap_mismatch|TPR"], "FAR_by": m["swap_mismatch|FAR_by"], "FAR_orig": m["FAR_orig"],
                      "DiD_mis": m["swap_mismatch|DiD"] * 100, "DiD_typed": m["swap_typed|DiD"] * 100,
                      "DiD_llm": m["llm_rewrite|DiD"] * 100})
    print("Single-passage runs (tested on mismatch fakes; DiD in points)\n")
    print(t.round({"TPR": 3, "FAR_by": 3, "FAR_orig": 3, "DiD_mis": 1, "DiD_typed": 1, "DiD_llm": 1}).to_string(index=False))


def xdoc():
    a = pd.read_csv(ROOT / "results" / "xdoc_auc.csv")
    a = a[a.train == "swap_typed"].pivot_table(index=["ref", "group"], columns="test", values="AUC_orig")
    print("Same-type-trained model, AUROC fake vs original\n")
    print(a[["swap_typed", "llm_rewrite"]].rename(columns={"swap_typed": "same-type", "llm_rewrite": "LLM"}).round(3).to_string())


def example():
    for l in open(ROOT / "data" / "sets" / "test_squad.jsonl"):
        d = json.loads(l)
        if d["rid"] == "56de4a89cffd8e1900b4b7bd" and d["gen"] == "swap_mismatch" and d["kind"] in ("fake", "bystander"):
            d["p"] = d["p"][:160] + " ..."
            print(json.dumps(d, indent=1, ensure_ascii=False), "\n")


if __name__ == "__main__":
    {"training": training, "main": main_table, "xdoc": xdoc, "example": example}[sys.argv[1]]()
