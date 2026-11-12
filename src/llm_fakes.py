"""Stage 2: the `llm_rewrite` generator.

Qwen2.5-1.5B-Instruct (frozen, greedy) rewrites each passage so that it states NEW instead of OLD, adjusting
nearby wording for fluency/consistency. Applied to
  - the answer   -> misleading fake  (same fake answer as swap_typed, so only the fabrication process differs)
  - a non-answer -> placebo          (same process, passage still supports the question)
Outputs are validated (NEW present, OLD absent, answer kept/removed as required, length ratio sane).
Resumable: appends to data/llm_{split}.jsonl and skips finished jobs.
"""
import json, argparse, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = ('Edit the passage so that it states "{new}" wherever it currently states "{old}". Adjust nearby wording '
          'so the passage stays fluent and self-consistent, but keep all other facts unchanged. '
          'Output only the edited passage.\n\nPassage: {ctx}')
# retry prompt for failed rewrites (mostly: OLD left in somewhere, or passage copied unchanged)
RETRY = ('Rewrite the passage below so that "{old}" is replaced by "{new}" EVERYWHERE it appears. '
         'Important: the text "{old}" must not appear anywhere in your output. Keep the passage fluent and '
         'self-consistent and keep all other facts the same. Output only the rewritten passage.\n\nPassage: {ctx}')


def occurs(text, s):
    return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, flags=re.I) is not None


def valid(job, out, rec):
    if not out or not 0.7 <= len(out) / len(rec["context"]) <= 1.4:
        return False
    if not occurs(out, job["new"]) or occurs(out, job["old"]):
        return False
    if job["kind"] == "fake":
        return not occurs(out, rec["answer"])
    return occurs(out, rec["answer"])  # placebo must still contain the answer


def main(split, bs, max_records, kinds, retry=False):
    recs = [json.loads(l) for l in open(DATA / f"records_{split}.jsonl")]
    if max_records:  # keep the shortest passages (same length profile as the finished test set; faster)
        recs = sorted(recs, key=lambda r: len(r["context"]))[:max_records]
    recs = {r["id"]: r for r in recs}
    out_path = DATA / f"llm_{split}.jsonl"
    done = set()
    if out_path.exists():
        done = {(d["id"], d["kind"]) for d in map(json.loads, open(out_path))}
    jobs = []
    if retry:  # redo every finished job that fails the relaxed check used downstream, and has no valid attempt yet
        import sys; sys.path.insert(0, str(ROOT / "src"))
        from make_sets import llm_ok
        ok = set()
        prev = list(map(json.loads, open(out_path)))
        for d in prev:
            if d["id"] in recs and llm_ok(d, recs[d["id"]]):
                ok.add((d["id"], d["kind"]))
        seen = set()
        for d in prev:
            key = (d["id"], d["kind"])
            if d["id"] in recs and d["kind"] in kinds and key not in ok and key not in seen:
                seen.add(key)
                jobs.append({k: d[k] for k in ["id", "kind", "old", "new"]})
        done = set()
    for r in ([] if retry else recs.values()):
        for kind, e in [("fake", r["edits"]["fake"]), ("placebo", r["edits"]["placebo"])]:
            if kind in kinds and (r["id"], kind) not in done:
                jobs.append({"id": r["id"], "kind": kind, "old": e["old"], "new": e["typed"]})
    jobs.sort(key=lambda j: len(recs[j["id"]]["context"]))  # length-sort to reduce padding
    print(f"[{split}] {len(jobs)} jobs to run ({len(done)} done)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("mps").eval()
    n_ok = 0
    torch.manual_seed(0)  # reproducible sampling in the retry pass
    with open(out_path, "a") as f:
        for i in range(0, len(jobs), bs):
            batch = jobs[i:i + bs]
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": (RETRY if retry else PROMPT).format(new=j["new"], old=j["old"], ctx=recs[j["id"]]["context"])}],
                tokenize=False, add_generation_prompt=True) for j in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to("mps")
            max_new = int(enc.input_ids.shape[1] * 0.75) + 40
            with torch.no_grad():
                gen = m.generate(**enc, max_new_tokens=max_new, do_sample=retry, temperature=0.7 if retry else None,
                                 top_p=0.9 if retry else None)
            texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for j, t in zip(batch, texts):
                t = t.strip().removeprefix("Passage:").strip()
                ok = valid(j, t, recs[j["id"]])
                n_ok += ok
                f.write(json.dumps({**j, "text": t, "valid": ok, "attempt": 2 if retry else 1}) + "\n")
            f.flush()
            if (i // bs) % 10 == 0:
                print(f"[{split}] {i + len(batch)}/{len(jobs)} valid so far {n_ok}", flush=True)
    print(f"[{split}] done, valid {n_ok}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev,train")
    ap.add_argument("--bs", type=int, default=8)          # small batch = low memory, laptop stays usable
    ap.add_argument("--kinds", default="fake")            # train/dev only need fakes; test already has both
    ap.add_argument("--max_records", default="dev:300,train:1500")
    ap.add_argument("--retry", action="store_true")
    a = ap.parse_args()
    caps = dict(x.split(":") for x in a.max_records.split(",")) if a.max_records else {}
    for s in a.splits.split(","):
        main(s, a.bs, int(caps.get(s, 0)), a.kinds.split(","), a.retry)
