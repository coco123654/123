#!/usr/bin/env python3
"""
Generate GO enrichment dot plot for 20-day symbiotic-specific genes
using elimKS results from topGO analysis.

Reads the pre-computed elimKS GO enrichment results and creates a
publication-quality dot plot with:
  - Top GO terms per ontology (BP, MF, CC)
  - Dot size  = number of significant genes
  - Dot color = -log10(elimKS p-value)
  - X-axis    = Rich Factor (Significant / Annotated)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GO_ALL_FILE = os.path.join(DATA_DIR, "GO_enrichment_elimKS_all.csv")
GENE_LIST_FILE = os.path.join(DATA_DIR, "20day_specific_genes.csv")
OUT_PNG = os.path.join(DATA_DIR, "GO_enrichment_20day_elimKS.png")
OUT_PDF = os.path.join(DATA_DIR, "GO_enrichment_20day_elimKS.pdf")

# ── Configuration ──────────────────────────────────────────────────────────
TOP_N_PER_CATEGORY = 10          # top terms per ontology
MAX_TERM_LABEL_LEN = 55          # truncate long GO term names

# ── 1. Load data ──────────────────────────────────────────────────────────
df = pd.read_csv(GO_ALL_FILE)
print(f"Loaded {len(df)} GO terms from {GO_ALL_FILE}")

# Load gene list summary
gene_df = pd.read_csv(GENE_LIST_FILE)
n_up = (gene_df["Category"] == "Symbiotic_specific_UP").sum()
n_down = (gene_df["Category"] == "NonSymbiotic_specific_DOWN").sum()
n_total = len(gene_df)
print(f"Target genes: {n_total}  (Symbiotic UP: {n_up}, Non-symbiotic DOWN: {n_down})")

# ── 2. Compute derived columns ────────────────────────────────────────────
df["Rich_Factor"] = df["Significant"] / df["Annotated"]
df["neg_log10_p"] = -np.log10(df["elimKS_pvalue"].clip(lower=1e-20))

# ── 3. Select top terms per ontology ──────────────────────────────────────
ontology_order = ["BP", "MF", "CC"]
ontology_labels = {
    "BP": "Biological Process",
    "MF": "Molecular Function",
    "CC": "Cellular Component",
}

frames = []
for ont in ontology_order:
    sub = df[df["Ontology"] == ont].sort_values("elimKS_pvalue").head(TOP_N_PER_CATEGORY)
    frames.append(sub)

plot_df = pd.concat(frames, ignore_index=True)
# Reverse so that top-ranked terms per category appear at the top when plotted
plot_df = plot_df.iloc[::-1].reset_index(drop=True)
print(f"Selected {len(plot_df)} GO terms for the figure")

# Truncate long term names
def truncate(s, maxlen=MAX_TERM_LABEL_LEN):
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."

plot_df["Label"] = plot_df["GO_Term"].apply(truncate)

# ── 4. Build figure ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.38)))

# Colour mapping: -log10(p-value), higher = more significant
cmap = plt.cm.RdYlBu_r       # warm colours for significant terms
norm = Normalize(vmin=0, vmax=plot_df["neg_log10_p"].max() * 1.05)

# Size mapping: number of significant genes
size_scale = 30               # base multiplier
sizes = plot_df["Significant"].values * size_scale
sizes = np.clip(sizes, 30, 800)  # enforce min/max bubble size

# Colour values
colours = plot_df["neg_log10_p"].values

scatter = ax.scatter(
    plot_df["Rich_Factor"],
    np.arange(len(plot_df)),
    s=sizes,
    c=colours,
    cmap=cmap,
    norm=norm,
    edgecolors="black",
    linewidths=0.6,
    zorder=3,
    alpha=0.9,
)

# Y-axis labels
ax.set_yticks(np.arange(len(plot_df)))
ax.set_yticklabels(plot_df["Label"], fontsize=9)

# Add ontology colour bands on the left
ont_colours = {"BP": "#D5E8D4", "MF": "#DAE8FC", "CC": "#FFF2CC"}
prev_ont = None
band_start = 0
# Group contiguous ontology blocks
ontologies_in_order = plot_df["Ontology"].values
for i in range(len(ontologies_in_order)):
    if i == 0:
        prev_ont = ontologies_in_order[i]
        band_start = i
        continue
    if ontologies_in_order[i] != prev_ont:
        # Draw previous band
        ax.axhspan(band_start - 0.5, i - 0.5, color=ont_colours.get(prev_ont, "white"),
                    alpha=0.25, zorder=0)
        mid = (band_start + i - 1) / 2
        ax.text(-0.005, mid, ontology_labels.get(prev_ont, prev_ont),
                ha="right", va="center", fontsize=9, fontweight="bold",
                color="#333333", transform=ax.get_yaxis_transform())
        prev_ont = ontologies_in_order[i]
        band_start = i
# Last band
ax.axhspan(band_start - 0.5, len(ontologies_in_order) - 0.5,
           color=ont_colours.get(prev_ont, "white"), alpha=0.25, zorder=0)
mid = (band_start + len(ontologies_in_order) - 1) / 2
ax.text(-0.005, mid, ontology_labels.get(prev_ont, prev_ont),
        ha="right", va="center", fontsize=9, fontweight="bold",
        color="#333333", transform=ax.get_yaxis_transform())

# Axis labels & grid
ax.set_xlabel("Rich Factor (Significant / Annotated)", fontsize=12, fontweight="bold")
ax.set_title(
    "GO Enrichment Analysis (elimKS) — 20-Day Symbiotic-Specific Genes",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
ax.set_xlim(left=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Colour bar (p-value) ──────────────────────────────────────────────────
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02, aspect=20)
cbar.set_label("$-\\log_{10}$(elimKS p-value)", fontsize=10)

# ── Size legend ───────────────────────────────────────────────────────────
# Pick a few representative gene counts for the legend
sig_vals = sorted(plot_df["Significant"].unique())
if len(sig_vals) > 5:
    legend_vals = [sig_vals[0], sig_vals[len(sig_vals) // 3],
                   sig_vals[2 * len(sig_vals) // 3], sig_vals[-1]]
else:
    legend_vals = sig_vals
legend_vals = sorted(set(legend_vals))

legend_handles = []
for v in legend_vals:
    s = np.clip(v * size_scale, 30, 800)
    h = ax.scatter([], [], s=s, c="gray", edgecolors="black", linewidths=0.5)
    legend_handles.append(h)

legend_labels_text = [str(int(v)) for v in legend_vals]
leg = ax.legend(
    legend_handles,
    legend_labels_text,
    title="Significant\nGenes",
    loc="lower right",
    fontsize=9,
    title_fontsize=9,
    frameon=True,
    labelspacing=1.5,
    borderpad=1.2,
    handletextpad=1.5,
)
leg.get_frame().set_edgecolor("gray")

# ── Subtitle with gene count info ─────────────────────────────────────────
fig.text(
    0.5,
    0.01,
    f"Target genes: {n_total}  |  Symbiotic-specific UP (MS20): {n_up}  |  "
    f"Non-symbiotic-specific DOWN (M20): {n_down}",
    ha="center",
    fontsize=9,
    fontstyle="italic",
    color="#555555",
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\n✓ Figure saved: {OUT_PNG}")
print(f"✓ Figure saved: {OUT_PDF}")
plt.close()
