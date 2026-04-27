"""Generate SHAP importance bar chart for all three dataset families."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(BASE, "results", "shap_importance.csv")
OUT  = os.path.join(BASE, "results", "shap_importance_fig.pdf")

FAMILY_LABELS = {
    "promise-ck": "PROMISE CK",
    "aeeem":      "AEEEM",
    "nasa":       "NASA MDP",
}

FEATURE_LABELS = {
    # AEEEM — raw names end with trailing underscore
    "agewithrespectto_":      "age w.r.t. first change",
    "linesaddeduntil_":       "lines added",
    "maxlinesremoveduntil_":  "max lines removed",
    "weightedagewithrespectto_": "weighted age",
    "numberofversionsuntil_": "num versions",
    # NASA
    "loc_total":      "loc total",
    "loc_executable": "loc executable",
    "loc_blank":      "loc blank",
    "loc_comments":   "loc comments",
    "halstead_content": "halstead content",
}

df = pd.read_csv(SRC)

families = ["promise-ck", "aeeem", "nasa"]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))
fig.subplots_adjust(wspace=0.55, left=0.01, right=0.99, top=0.88, bottom=0.18)

BAR_COLOR = "#3a7abf"

for ax, family in zip(axes, families):
    fam = df[df["family"] == family].head(5).sort_values("mean_abs_shap", ascending=True)
    raw_names = fam["feature"].tolist()
    labels    = [FEATURE_LABELS.get(n, n.replace("_", " ")) for n in raw_names]
    vals      = fam["mean_abs_shap"].tolist()

    bars = ax.barh(range(len(labels)), vals, color=BAR_COLOR, edgecolor="none", height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_title(FAMILY_LABELS[family], fontsize=8, fontweight="bold", pad=3)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("Mean |SHAP|", fontsize=7, labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # value labels on bars
    for bar, v in zip(bars, vals):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=5.5, color="#333333")

    ax.set_xlim(0, max(vals) * 1.30)

plt.savefig(OUT, bbox_inches="tight")
print(f"Saved: {OUT}")
