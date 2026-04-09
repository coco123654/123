#!/usr/bin/env python3
"""
Combined Multi-group Differential Volcano Plot
Non-symbiotic (M) on left, Symbiotic (MS) on right
Time points: 10, 20, 30, 40, 50 (vs M5 baseline)
Center labels show only time numbers (10, 20, 30, 40, 50).
Style: reference image with colored center bands, per-group colored dots,
       alternating grey bands, and clear section headers.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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

# ── Colors ─────────────────────────────────────────────────────────────────
# Each group gets a unique color for both center band and significant dots
GROUP_COLORS = {
    "M10":  "#FF6347",   # tomato
    "M20":  "#FFA500",   # orange
    "M30":  "#FFD700",   # gold
    "M40":  "#32CD32",   # lime green
    "M50":  "#6495ED",   # cornflower blue
    "MS10": "#E84393",   # pink
    "MS20": "#DA70D6",   # orchid
    "MS30": "#00CED1",   # dark turquoise
    "MS40": "#4169E1",   # royal blue
    "MS50": "#9370DB",   # medium purple
}

# Time labels to display in center bands (just the number)
TIME_LABELS = {
    "M10": "10", "M20": "20", "M30": "30", "M40": "40", "M50": "50",
    "MS10": "10", "MS20": "20", "MS30": "30", "MS40": "40", "MS50": "50",
}

NONSIG_COLOR = "#CCCCCC"   # grey for non-significant
UPREG_COLOR = "#E74C3C"    # red for up-regulated annotation
DOWNREG_COLOR = "#3498DB"  # blue for down-regulated annotation


# ── Read all data ──────────────────────────────────────────────────────────
def read_deg(filename, label):
    """Read a DEG CSV and return a tidy DataFrame."""
    path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    # Columns: 0=GeneID, 1=GeneName, 2=GeneDesc, 3=FC, 4=Log2FC,
    #          5=Pvalue, 6=Padjust, 7=Significant, 8=Regulate
    col_map = {
        cols[0]: "GeneID",
        cols[1]: "GeneName",
        cols[2]: "GeneDesc",
        cols[3]: "FC",
        cols[4]: "Log2FC",
        cols[5]: "Pvalue",
        cols[6]: "Padjust",
        cols[7]: "Significant",
        cols[8]: "Regulate",
    }
    df = df.rename(columns=col_map)
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
fig, ax = plt.subplots(figsize=(20, 9))

# Alternating grey background bands
for i in range(n_groups):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#F5F5F5", zorder=0)

# Vertical separator between M and MS blocks
ax.axvline(x=4.5, color="grey", linewidth=1.5, linestyle="--", zorder=1, alpha=0.6)

# Horizontal line at y=0 (behind center band)
ax.axhline(y=0, color="grey", linewidth=0.5, zorder=1)

# ── Plot points (per-group coloring) ──────────────────────────────────────
for grp in group_order:
    grp_data = data[data["label"] == grp]
    sig = grp_data[grp_data["is_sig"]]
    nonsig = grp_data[~grp_data["is_sig"]]

    # Non-significant – grey
    ax.scatter(
        nonsig["x"], nonsig["Log2FC"],
        c=NONSIG_COLOR, s=10, alpha=0.35, edgecolors="none", zorder=2,
    )
    # Significant – group color
    ax.scatter(
        sig["x"], sig["Log2FC"],
        c=GROUP_COLORS[grp], s=16, alpha=0.75, edgecolors="none", zorder=3,
    )

# ── Center color bands with TIME-ONLY labels ──────────────────────────────
band_height = 1.5  # half-height of center label band
for i, grp in enumerate(group_order):
    color = GROUP_COLORS[grp]
    time_label = TIME_LABELS[grp]

    # Draw colored rounded rectangle at y=0
    rect = FancyBboxPatch(
        (i - 0.42, -band_height),
        0.84,
        2 * band_height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.92,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        i, 0, time_label,
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="white",
        zorder=6,
    )

# ── Axes styling ───────────────────────────────────────────────────────────
ax.set_xlim(-0.7, n_groups - 0.3)

# Determine y limits symmetrically
y_abs_max = data["Log2FC"].abs().quantile(0.995)
y_abs_max = max(y_abs_max, 6)
ax.set_ylim(-y_abs_max * 1.18, y_abs_max * 1.18)

ax.set_ylabel("log₂FoldChange", fontsize=14, fontweight="bold")
ax.set_xlabel("")

# x-axis: no default tick labels
ax.set_xticks(range(n_groups))
ax.set_xticklabels([""] * n_groups)

# Section headers: "Non-symbiotic (M)" and "Symbiotic (MS)"
n_m = len(m_files)
n_ms = len(ms_files)
m_center = (n_m - 1) / 2
ms_center = n_m + (n_ms - 1) / 2

ax.text(
    m_center, -y_abs_max * 1.10,
    "Non-symbiotic (M)",
    ha="center", va="top", fontsize=13, fontweight="bold", color="#555",
)
ax.text(
    ms_center, -y_abs_max * 1.10,
    "Symbiotic (MS)",
    ha="center", va="top", fontsize=13, fontweight="bold", color="#555",
)

# Up-regulated / Down-regulated side annotations
ax.text(
    -0.55, y_abs_max * 0.65,
    "Up-regulated ↑",
    ha="left", va="center", fontsize=11, fontweight="bold", color=UPREG_COLOR,
    rotation=90, zorder=8,
)
ax.text(
    -0.55, -y_abs_max * 0.65,
    "Down-regulated ↓",
    ha="left", va="center", fontsize=11, fontweight="bold", color=DOWNREG_COLOR,
    rotation=90, zorder=8,
)

# ── Legend ──────────────────────────────────────────────────────────────────
handles = []
for grp in group_order:
    handles.append(
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=GROUP_COLORS[grp],
                   markersize=7, label=grp)
    )
handles.append(
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=NONSIG_COLOR,
               markersize=7, label='Not Sig')
)
ax.legend(handles=handles, loc="upper right", fontsize=8.5,
          framealpha=0.9, ncol=1, title="Group", title_fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Title
ax.set_title(
    "Combined Multi-group Differential Volcano Plot\n"
    "(Non-symbiotic M vs Symbiotic MS, Time: 10–50 d)",
    fontsize=15, fontweight="bold", pad=18,
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
