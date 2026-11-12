"""Where does the detector look?  Gradient x input attribution of the MISLEADING logit.

For each test record and generator we take the fake passage under (a) the original question and (b) the bystander
question. Since the passage text is identical, any difference in attribution is caused by the question alone.
We report the share of |attribution| that falls on the inserted fake entity (span of the NEW string in the passage).
A content-sensitive detector should focus on the inserted span only when the question asks about it.

  python src/attribution.py --run swap_typed_base_s0 --n 300
"""
import json, argparse, re
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
GENS = ["swap_mismatch", "swap_typed", "llm_rewrite"]


def span_share(model, tok, q, p, new, max_len=384):
    enc = tok(q, p, truncation="only_second", max_length=max_len, return_offsets_mapping=True, return_tensors="pt")
    offs, seq = enc.pop("offset_mapping")[0].tolist(), enc.sequence_ids(0)
    m = re.search(r"(?<!\w)" + re.escape(new) + r"(?!\w)", p, flags=re.I)
    if m is None:
        return None
    enc = enc.to(DEV)
    emb = model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
    out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"], token_type_ids=enc.get("token_type_ids"))
    out.logits[0, 1].backward()
    attr = (emb.grad[0] * emb[0]).sum(-1).abs().detach().cpu().numpy()
    pas = [i for i, s in enumerate(seq) if s == 1]
    in_span = [i for i in pas if offs[i][0] < m.end() and offs[i][1] > m.start()]
    if not in_span:
        return None
    tot = attr[pas].sum()
    return float(attr[in_span].sum() / tot), len(in_span) / len(pas)


def main(a):
    mdir = ROOT / "runs" / a.run / "model"
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir, dtype=torch.float32).to(DEV).eval()
    import sys; sys.path.insert(0, str(ROOT / "src"))
    from make_sets import load_split
    recs = load_split("test")[: a.n]
    res = {}
    for g in GENS:
        rows = []
        for r in recs:
            new = r["edits"]["fake"]["typed" if g != "swap_mismatch" else "mismatch"]
            p = r["passages"][g]["fake"]
            fq = span_share(model, tok, r["question"], p, new)
            fb = span_share(model, tok, r["bystander_q"], p, new)
            if fq and fb:
                rows.append((fq[0], fb[0], fq[1]))
        x = np.array(rows)
        res[g] = {"n": len(x), "share_fake_q": float(x[:, 0].mean()), "share_bystander_q": float(x[:, 1].mean()),
                  "span_token_frac": float(x[:, 2].mean()),
                  "ratio_q_over_by": float(x[:, 0].mean() / x[:, 1].mean())}
        print(g, res[g], flush=True)
    out = ROOT / "results" / f"attribution_{a.run}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=300)
    main(ap.parse_args())
