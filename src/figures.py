"""Report figures -> results/figures/*.png (reads results/*.csv; run after analyze.py, analyze_xdoc.py)."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a"]          # fixed categorical order
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
GEN = ["swap_mismatch", "swap_typed", "llm_rewrite"]
GL = {"swap_mismatch": "Mismatch swap", "swap_typed": "Same-type swap", "llm_rewrite": "LLM rewrite"}


def bars(ax, groups, series, vals, fmt="{:.0f}", ylim=None):
    w = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        b = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals[i], w * 0.92, color=C[i], label=s, zorder=3)
        for r, v in zip(b, vals[i]):
            ax.text(r.get_x() + r.get_width() / 2, v + (0.01 if v >= 0 else -0.01) * (ylim[1] if ylim else 1),
                    fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
    ax.set_xticks(x, groups)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    if ylim:
        ax.set_ylim(*ylim)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.set_xlim(0, 104); ax.set_ylim(0, 38); ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f6fb", ec="#2a78d6", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    box(0.5, 14, 14, 9, "SQuAD v1.1\nquestion, answer,\nparagraph")
    gens = [("Mismatch swap\nanswer → other type", 27), ("Same-type swap\nanswer → same type", 15),
            ("LLM rewrite\nQwen2.5-1.5B", 3)]
    for t, y in gens:
        box(19, y, 19, 8, t, fc="#fdf1ec", ec="#eb6834")
        arrow(15, 18.5, 18.6, y + 4)
    box(43, 13, 22, 11, "Model input\n[CLS] question [SEP]\n(Reference: snippet [SEP])\ncandidate passage [SEP]")
    for _, y in gens:
        arrow(38.4, y + 4, 42.6, 18.5)
    box(69.5, 13, 15, 11, "DeBERTa-v3-small\nNLI cross-encoder\n(fine-tuned)", fc="#eaf7f1", ec="#1baf7a", bold=True)
    arrow(65.4, 18.5, 69.1, 18.5)
    box(89, 13, 14, 11, "Supporting\nMisleading\nIrrelevant")
    arrow(84.9, 18.5, 88.6, 18.5)
    box(43, 29, 22, 7, "Other paragraphs of the\nsame article (Section 5)", fc="#ffffff", ec=MUTED)
    arrow(54, 28.7, 54, 24.4)
    box(43, 1, 60, 8, "Diagnostics: bystander test, placebo edit, question-blind probe,\n"
        "2×2 difference-in-differences (DiD), gradient×input attribution", fc="#ffffff", ec=MUTED)
    arrow(77, 12.6, 77, 9.4)
    fig.savefig(OUT / "fig1_pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_baseline(m):
    r = m.loc[("swap_mismatch", "base")]
    vals = [[r[f"{g}|TPR"] * 100 for g in GEN], [r[f"{g}|FAR_by"] * 100 for g in GEN], [r["FAR_orig"] * 100] * 3]
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    bars(ax, [GL[g] for g in GEN], ["Fake passage, its own question (should flag)",
                                    "Same fake passage, untouched question (should NOT flag)",
                                    "Original passage (should NOT flag)"], vals, ylim=(0, 80))
    ax.set_ylabel("% flagged misleading")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(OUT / "fig2_baseline_bystander.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_did(m):
    rows = [("swap_mismatch", "base", "Mismatch\nbase"), ("swap_mismatch", "pair", "Mismatch\npair (ours)"),
            ("swap_mismatch", "placebo", "Mismatch\nplacebo"), ("swap_mismatch", "qblind", "Mismatch\nq-blind"),
            ("mix", "base", "Mix\nbase"), ("mix", "pair", "Mix\npair"), ("swap_typed", "base", "Same-type\nbase"),
            ("llm_rewrite", "base", "LLM\nbase")]
    vals = [[m.loc[(g, r)][f"{t}|DiD"] * 100 for g, r, _ in rows] for t in GEN]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    bars(ax, [l for *_, l in rows], ["Tested on " + GL[t] for t in GEN], vals, ylim=(-8, 24))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("DiD (points)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(OUT / "fig3_did_regimes.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_xdoc():
    a = pd.read_csv(ROOT / "results" / "xdoc_auc.csv")
    a = a[a.train == "swap_typed"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, tg in zip(axes, ["swap_typed", "llm_rewrite"]):
        get = lambda ref, grp: a[(a.ref == ref) & (a.group == grp) & (a.test == tg)].AUC_orig.item()
        vals = [[get("none", g) for g in ["corr", "nocorr"]], [get("good", g) for g in ["corr", "nocorr"]],
                [get("keyword", g) for g in ["corr", "nocorr"]]]
        bars(ax, ["Reference contains\nthe true answer", "Reference does not"],
             ["No reference (single passage)", "With reference (good retriever)", "With reference (keyword retriever)"],
             vals, fmt="{:.2f}", ylim=(0.4, 0.8))
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_title("Tested on " + GL[tg], fontsize=9, color=INK)
    axes[0].set_ylabel("AUROC (fake vs original)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(OUT / "fig4_xdoc_auroc.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    m = pd.read_csv(ROOT / "results" / "mean.csv").set_index(["train_gen", "regime"])
    fig_pipeline(); fig_baseline(m); fig_did(m); fig_xdoc()
    print(sorted(p.name for p in OUT.iterdir()))
