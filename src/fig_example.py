"""Figure: one real test record (dataset variants) and how the classifier reacts to it.
Word shading = gradient x input attribution of the 'misleading' logit (as in attribution.py), normalised per row.
  python src/fig_example.py   ->  results/figures/fig_example.png
"""
import json, re, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from make_sets import load_split
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
RID = "56de4a89cffd8e1900b4b7bd"            # Norman architecture record (SQuAD validation)
SHOW = ("They spread", "proportions.")       # passage window shown in panel (b)
plt.rcParams.update({"font.family": "Arial"})
INK, MUTED, EDIT = "#222222", "#666666", "#eb6834"


def word_attr(model, tok, q, p):
    enc = tok(q, p, truncation="only_second", max_length=384, return_offsets_mapping=True, return_tensors="pt")
    offs, seq = enc.pop("offset_mapping")[0].tolist(), enc.sequence_ids(0)
    enc = enc.to(DEV)
    emb = model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
    out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"], token_type_ids=enc.get("token_type_ids"))
    pm = torch.softmax(out.logits[0], -1)[1].item()
    out.logits[0, 1].backward()
    a = (emb.grad[0] * emb[0]).sum(-1).abs().detach().cpu().numpy()
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", p)]
    wa = np.zeros(len(words))
    for i, s in enumerate(seq):
        if s != 1:
            continue
        for j, (ws, we) in enumerate(words):
            if offs[i][0] < we and offs[i][1] > ws:
                wa[j] += a[i]; break
    return [p[s:e] for s, e in words], wa, pm


def draw_words(ax, x, y, words, shade, edit_idx, fs=8):
    """One line of words; background opacity = attribution (only the strongest words get visible colour)."""
    r = ax.figure.canvas.get_renderer()
    for i, (w, a) in enumerate(zip(words, shade)):
        fc = (0.165, 0.47, 0.84, 0.9 * a) if a > 0 else (1, 1, 1, 0)
        t = ax.text(x, y, w, fontsize=fs, va="center", ha="left", color="white" if a > 0.6 else INK,
                    fontweight="bold" if i in edit_idx else "normal",
                    bbox=dict(boxstyle="square,pad=0.15", fc=fc, ec="none"))
        x = t.get_window_extent(r).transformed(ax.transData.inverted()).x1 + 0.007


def main():
    rec = next(r for r in load_split("test") if r["id"] == RID)
    q, qb = rec["question"], rec["bystander_q"]
    fake = rec["passages"]["swap_mismatch"]["fake"]
    vals = {}
    for run in ["swap_mismatch_base_s0", "swap_mismatch_pair_s0"]:
        tok = AutoTokenizer.from_pretrained(ROOT / "runs" / run / "model")
        model = AutoModelForSequenceClassification.from_pretrained(ROOT / "runs" / run / "model", dtype=torch.float32).to(DEV).eval()
        for k, qq in [("q", q), ("qby", qb)]:
            w, a, pm = word_attr(model, tok, qq, fake)
            share = a / a.sum()
            vals[(run, k)] = (pm, sum(s_ for ww, s_ in zip(w, share) if ww.strip(",.") == "12,843"), 1 / len(w))
        del model
    json.dump({f"{r}|{k}": v for (r, k), v in vals.items()}, open(ROOT / "results" / "example_values.json", "w"), indent=1)

    fig = plt.figure(figsize=(7.4, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.5], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    T = lambda x, y, t, **k: ax.text(x, y, t, va="center", color=k.pop("color", INK), **k)
    T(0.0, 1.0, "(a) One test record", fontsize=9.5, fontweight="bold")
    rows = [("Question q", q + "  → Romanesque", None), ("Bystander q_by", qb + "  → rounded", None),
            ("Mismatch fake", "“They spread a unique ", "12,843"), ("Same-type fake", "“They spread a unique ", "Life Itself Roger Ebert"),
            ("LLM fake", "“They spread a unique ", "Life Itself")]
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for i, (k, v, e) in enumerate(rows):
        y = 0.78 - i * 0.19
        T(0.0, y, k, fontsize=8, fontweight="bold")
        t = T(0.2, y, v, fontsize=8)
        if e:
            x = t.get_window_extent(r).transformed(ax.transData.inverted()).x1
            t2 = T(x, y, e, fontsize=8, fontweight="bold", color=EDIT)
            x = t2.get_window_extent(r).transformed(ax.transData.inverted()).x1
            T(x, y, " idiom to England and Italy …”", fontsize=8)

    labels = ["Baseline", "Pair training\n(ours)"]
    runs = ["swap_mismatch_base_s0", "swap_mismatch_pair_s0"]
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    xs = np.arange(2); wd = 0.36; handles = []
    for j, (title, idx, ylab, fmt) in enumerate([
            ("(b) Prediction on the mismatch fake", 0, "P(misleading)", "{:.2f}"),
            ("(c) Attribution on the edited word “12,843”", 1, "Share of attribution (%)", "{:.0f}%")]):
        a2 = fig.add_subplot(gs[1, j])
        for i, (k, lab, c) in enumerate([("q", "Asked q (should flag)", BLUE), ("qby", "Asked q_by (should NOT flag)", ORANGE)]):
            v = [vals[(rr, k)][idx] * (100 if idx else 1) for rr in runs]
            b = a2.bar(xs + (i - 0.5) * wd, v, wd * 0.92, color=c, label=lab, zorder=3)
            for rect, vv in zip(b, v):
                a2.text(rect.get_x() + rect.get_width() / 2, vv, fmt.format(vv), ha="center", va="bottom", fontsize=7.5, color=INK)
        a2.set_xticks(xs, labels, fontsize=8)
        a2.set_title(title, fontsize=9, fontweight="bold", loc="left", color=INK)
        a2.set_ylabel(ylab, fontsize=8)
        a2.tick_params(labelsize=7.5)
        a2.spines[["top", "right"]].set_visible(False)
        a2.yaxis.grid(True, color="#e3e3e3", zorder=0)
        if idx == 0:
            a2.set_ylim(0, 1.12); a2.axhline(0.5, color=MUTED, ls="--", lw=0.8)
            a2.plot([], [], color=MUTED, ls="--", lw=0.8, label="Flag threshold (0.5)")
        else:
            avg = np.mean([vals[(rr, k)][2] for rr in runs for k in ["q", "qby"]]) * 100
            a2.set_ylim(0, 34); a2.axhline(avg, color=MUTED, ls="--", lw=0.8)
            a2.plot([], [], color=MUTED, ls=":", lw=0.8, label=f"Share of an average word ({avg:.1f}%)")
            a2.lines[0].set_linestyle(":")
        handles += [h for h in a2.get_legend_handles_labels()[0] if h.get_label() not in [x.get_label() for x in handles]]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    out = ROOT / "results" / "figures" / "fig_example.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    print(out, {k: tuple(round(x, 3) for x in v) for k, v in vals.items()})


if __name__ == "__main__":
    main()
