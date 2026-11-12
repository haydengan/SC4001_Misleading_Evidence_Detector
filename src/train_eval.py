"""Train one evidence classifier config and evaluate it on every test set.

  python src/train_eval.py --gen swap_typed --regime base --seed 0
  python src/train_eval.py --gen swap_typed --regime base --qblind   # question-blind probe

Model: microsoft/deberta-v3-small cross-encoder, input "[CLS] question [SEP] passage [SEP]" -> 3 classes.
Model selection uses the dev split (held-out SQuAD-train articles); test sets are never used for selection.
Writes runs/<name>/{metrics_dev.json, preds.jsonl, config.json} (+ model/ for seed 0).
"""
import json, argparse, random, time, os
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SETS = Path(os.environ.get("SETS_DIR", ROOT / "data" / "sets"))
RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE = os.environ.get("BACKBONE", "cross-encoder/nli-deberta-v3-small")  # NLI-pretrained; plain deberta-v3-small could not learn S vs M


def load(name):
    return [json.loads(l) for l in open(SETS / name)]


def make_collate(tok, qblind, max_len):
    def collate(batch):
        qs = ["" if qblind else b["q"] for b in batch]
        # pad to a multiple of 64: few distinct shapes, so the MPS allocator cache does not keep growing
        enc = tok(qs, [b["p"] for b in batch], truncation="only_second", max_length=max_len,
                  padding=True, pad_to_multiple_of=64, return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return enc
    return collate


@torch.no_grad()
def predict(model, rows, collate, bs=64):
    model.eval()
    # length-sort for speed, then restore order
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["p"]))
    probs = np.zeros((len(rows), 3), dtype=np.float32)
    for s in range(0, len(rows), bs):
        idx = order[s:s + bs]
        enc = collate([rows[i] for i in idx]).to(DEV)
        enc.pop("labels")
        probs[idx] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
    return probs


def main(a):
    name = f"{a.tag}{a.gen}_{'qblind' if a.qblind else a.regime}_s{a.seed}"
    out = RUNS / name
    if (out / "preds.jsonl").exists() and not a.force:
        print("exists, skip", name)
        return
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    regime = "base" if a.qblind else a.regime
    train, dev = load(f"train_{a.gen}_{regime}.jsonl"), load(f"dev_{a.gen}_{regime}.jsonl")
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=3, dtype=torch.float32,
                                                               ignore_mismatched_sizes=True).to(DEV)
    # freeze the 128k x 768 word-embedding table (98M of 141M params): fits 16 GB unified memory, faster,
    # and is a common choice when fine-tuning on a few thousand examples
    model.get_input_embeddings().weight.requires_grad_(False)
    collate = make_collate(tok, a.qblind, a.max_len)
    dl = DataLoader(train, batch_size=a.bs, shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(a.seed))
    # foreach=False: update params one by one (avoids big temporary copies; matters on 16 GB unified memory)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01, foreach=False)
    total = len(dl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    best, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for step, batch in enumerate(dl):
            # micro-batches of bs/accum: same effective batch, half the peak attention memory
            for mb in range(0, len(batch["labels"]), a.bs // a.accum):
                sub = {k: v[mb:mb + a.bs // a.accum].to(DEV) for k, v in batch.items()}
                loss = model(**sub).loss / a.accum
                loss.backward()
            loss = loss * a.accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if DEV == "mps" and step % 25 == 0:
                torch.mps.empty_cache()
            if step % 50 == 0:
                mem = torch.mps.current_allocated_memory() / 2**30 if DEV == "mps" else 0
                print(f"[{name}] ep{ep} step{step}/{len(dl)} loss {loss.item():.3f} {time.time() - t0:.0f}s mem {mem:.1f}G",
                      flush=True)
        p = predict(model, dev, collate)
        if DEV == "mps":
            torch.mps.empty_cache()
        f1 = f1_score([d["label"] for d in dev], p.argmax(1), average="macro")
        hist.append({"epoch": ep, "dev_macro_f1": float(f1)})
        print(f"[{name}] epoch {ep} dev macro-F1 {f1:.4f}", flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    test = load("test_squad.jsonl") + load("test_external.jsonl")
    p = predict(model, test, collate)
    with open(out / "preds.jsonl", "w") as f:
        for d, pr in zip(test, p):
            f.write(json.dumps({"uid": d["uid"], "gen": d["gen"], "kind": d["kind"], "type": d["type"],
                                "label": d["label"], "p": [round(float(x), 5) for x in pr]}) + "\n")
    json.dump({"history": hist, "best_dev_macro_f1": best, "train_time_s": time.time() - t0},
              open(out / "metrics_dev.json", "w"), indent=1)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    if a.seed == 0:
        model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=["swap_mismatch", "swap_typed", "llm_rewrite", "mix"])
    ap.add_argument("--regime", default="base", choices=["base", "pair", "placebo"])
    ap.add_argument("--qblind", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="", help="prefix for the run directory (e.g. diag_)")
    main(ap.parse_args())
