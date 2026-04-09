#!/usr/bin/env python3
"""
Combined Multi-group Differential Volcano Plot
Non-symbiotic (M) on left, Symbiotic (MS) on right
Time points: 10, 20, 30, 40, 50 (vs M5 baseline)
Style: reference image with colored center labels, alternating grey bands,
       red (Sig) / blue (Not Sig) dots, gene ID labels for top genes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from adjustText import adjust_text

# ── Setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ── File definitions ───────────────────────────────────────────────────────
# Non-symbiotic (M) files – left side
m_files = [
    ("M10VSM5.csv", "M10"),
    ("M20VSM5.csv", "M20"),
    ("M30VSM5.csv", "M30"),
    ("M40VSM5.csv", "M40"),
    ("M50VSM5.csv", "M50"),
]

# Symbiotic (MS) files – right side
ms_files = [
    ("MS10VSM5.csv", "MS10"),
    ("MS20VSM5.csv", "MS20"),
    ("MS30VSM5.csv", "MS30"),
    ("MS40VSM5.csv", "MS40"),
    ("MS50VSM5.csv", "MS50"),
]

all_files = m_files + ms_files

# Center-band colors (alternating for visual distinction)
LABEL_COLORS = [
    "#D4A017",  # M10  – golden
    "#5B9BD5",  # M20  – blue
    "#70AD47",  # M30  – green
    "#ED7D31",  # M40  – orange
    "#A855F7",  # M50  – purple
    "#D4A017",  # MS10 – golden
    "#5B9BD5",  # MS20 – blue
    "#70AD47",  # MS30 – green
    "#ED7D31",  # MS40 – orange
    "#A855F7",  # MS50 – purple
]

SIG_COLOR = "#E74C3C"       # red for significant
NONSIG_COLOR = "#3498DB"    # blue for not significant
TOP_GENES_PER_DIRECTION = 2  # number of top genes to label per direction per group


# ── Read all data ──────────────────────────────────────────────────────────
def read_deg(filename, label):
    """Read a DEG CSV and return a tidy DataFrame."""
    path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "GeneID",
        cols[1]: "GeneName",
        cols[2]: "GeneDesc",
        cols[3]: "FC",
        cols[4]: "Log2FC",
        cols[5]: "Pvalue",
        cols[6]: "Padjust",
        cols[7]: "Significant",
        cols[8]: "Regulate",
    })
    df["Log2FC"] = pd.to_numeric(df["Log2FC"], errors="coerce")
    df["Pvalue"] = pd.to_numeric(df["Pvalue"], errors="coerce")
    df = df.dropna(subset=["Log2FC", "Pvalue"])
    df = df[df["Pvalue"] > 0].copy()
    df["neg_log10p"] = -np.log10(df["Pvalue"])
    df["is_sig"] = df["Significant"].str.strip().str.lower() == "yes"
    df["label"] = label
    return df[["GeneID", "Log2FC", "Pvalue", "neg_log10p", "is_sig", "Regulate", "label"]]


frames = []
for fname, lbl in all_files:
    frames.append(read_deg(fname, lbl))
data = pd.concat(frames, ignore_index=True)

group_order = [lbl for _, lbl in all_files]
data["group_idx"] = data["label"].map({g: i for i, g in enumerate(group_order)})

# ── Jitter x positions ────────────────────────────────────────────────────
rng = np.random.default_rng(42)
data["x"] = data["group_idx"] + rng.uniform(-0.35, 0.35, size=len(data))

# ── Create figure ──────────────────────────────────────────────────────────
n_groups = len(group_order)
fig, ax = plt.subplots(figsize=(18, 8))

# Alternating grey background bands
for i in range(n_groups):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#F0F0F0", zorder=0)

# Vertical separator between M and MS blocks
ax.axvline(x=4.5, color="grey", linewidth=1.2, linestyle="--", zorder=1, alpha=0.5)

# ── Plot points ────────────────────────────────────────────────────────────
sig_mask = data["is_sig"]
# Not significant
ax.scatter(
    data.loc[~sig_mask, "x"],
    data.loc[~sig_mask, "Log2FC"],
    c=NONSIG_COLOR, s=12, alpha=0.55, edgecolors="none", zorder=2, label="Not Sig",
)
# Significant
ax.scatter(
    data.loc[sig_mask, "x"],
    data.loc[sig_mask, "Log2FC"],
    c=SIG_COLOR, s=14, alpha=0.70, edgecolors="none", zorder=3, label="Sig",
)

# ── Center color band with time labels ─────────────────────────────────────
band_height = 1.5  # half-height of center label band
for i, (grp, color) in enumerate(zip(group_order, LABEL_COLORS)):
    # Determine display label: show time number with VS M5
    time_num = grp.replace("MS", "").replace("M", "")
    display_label = f"{grp}\nVS M5"

    # Draw colored rectangle at y=0
    rect = FancyBboxPatch(
        (i - 0.40, -band_height),
        0.80,
        2 * band_height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        i, 0, display_label,
        ha="center", va="center",
        fontsize=7, fontweight="bold", color="white",
        zorder=6, linespacing=1.2,
    )

# ── Label top significant genes per group ──────────────────────────────────
texts = []
for i, grp in enumerate(group_order):
    grp_data = data[(data["label"] == grp) & (data["is_sig"])]
    if grp_data.empty:
        continue
    # Top 2 upregulated + top 2 downregulated by |Log2FC|
    up = grp_data[grp_data["Log2FC"] > band_height].nlargest(TOP_GENES_PER_DIRECTION, "Log2FC")
    down = grp_data[grp_data["Log2FC"] < -band_height].nsmallest(TOP_GENES_PER_DIRECTION, "Log2FC")
    top_genes = pd.concat([up, down])
    for _, row in top_genes.iterrows():
        t = ax.annotate(
            row["GeneID"],
            xy=(row["x"], row["Log2FC"]),
            fontsize=5.5,
            color="#333333",
            zorder=7,
            ha="center",
        )
        texts.append(t)

# ── Axes styling ───────────────────────────────────────────────────────────
ax.set_xlim(-0.6, n_groups - 0.4)

# Determine y limits symmetrically
y_abs_max = data["Log2FC"].abs().quantile(0.995)
y_abs_max = max(y_abs_max, 6)
ax.set_ylim(-y_abs_max * 1.15, y_abs_max * 1.15)

ax.set_ylabel("log₂FoldChange", fontsize=13)
ax.set_xlabel("")

# x-axis: no default ticks; we add group header text above
ax.set_xticks(range(n_groups))
ax.set_xticklabels([""] * n_groups)

# Add "Non-symbiotic (M)" and "Symbiotic (MS)" header labels
n_m = len(m_files)
n_ms = len(ms_files)
m_center = (n_m - 1) / 2
ms_center = n_m + (n_ms - 1) / 2
ax.text(
    m_center, -y_abs_max * 1.08,
    "Non-symbiotic (M)",
    ha="center", va="top", fontsize=12, fontweight="bold", color="#555",
)
ax.text(
    ms_center, -y_abs_max * 1.08,
    "Symbiotic (MS)",
    ha="center", va="top", fontsize=12, fontweight="bold", color="#555",
)

# Upregulated / Downregulated annotations
ax.text(
    -0.45, y_abs_max * 0.75,
    "Up-regulated ↑",
    ha="left", va="center", fontsize=10, fontweight="bold", color=SIG_COLOR,
    rotation=90, zorder=8,
)
ax.text(
    -0.45, -y_abs_max * 0.75,
    "Down-regulated ↓",
    ha="left", va="center", fontsize=10, fontweight="bold", color=SIG_COLOR,
    rotation=90, zorder=8,
)

# Horizontal line at y=0 (behind center band)
ax.axhline(y=0, color="grey", linewidth=0.5, zorder=1)

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SIG_COLOR,
               markersize=7, label='Sig'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NONSIG_COLOR,
               markersize=7, label='Not Sig'),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Title
ax.set_title(
    "Combined Multi-group Differential Volcano Plot (Non-symbiotic M vs Symbiotic MS, 10–50 d)",
    fontsize=14, fontweight="bold", pad=15,
)

plt.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "combined_multi_volcano.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

out_pdf = os.path.join(BASE_DIR, "combined_multi_volcano.pdf")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_pdf}")

plt.close()
