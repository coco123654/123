#!/usr/bin/env python3
"""
Combined Multi-group Differential Volcano Plot
Non-symbiotic (M) on left, Symbiotic (MS) on right
Time points: 10, 20, 30, 40, 50 (vs M5 baseline)
Center labels show only time numbers (10, 20, 30, 40, 50).
Style: reference image with colored center bands, alternating grey bands.
Upregulated genes → RED, Downregulated genes → BLUE, Non-significant → GREY.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

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
# Each group gets a unique color for the center band only
BAND_COLORS = {
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

# Dot colors by regulation direction
UP_COLOR = "#E74C3C"       # red for up-regulated
DOWN_COLOR = "#3498DB"     # blue for down-regulated
NONSIG_COLOR = "#CCCCCC"   # grey for non-significant


# ── Read all data ──────────────────────────────────────────────────────────
def read_deg(filename, label):
    """Read a DEG CSV and return a tidy DataFrame."""
    path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(path)
    cols = df.columns.tolist()
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
    df["is_up"] = df["Regulate"].str.strip().str.lower() == "up"
    df["is_down"] = df["Regulate"].str.strip().str.lower() == "down"
    df["label"] = label
    return df[["GeneID", "Log2FC", "Pvalue", "neg_log10p",
               "is_sig", "is_up", "is_down", "Regulate", "label"]]


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
fig, ax = plt.subplots(figsize=(22, 9))

# Alternating light grey background bands
for i in range(n_groups):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#F5F5F5", zorder=0)

# Vertical separator between Non-symbiotic and Symbiotic blocks
ax.axvline(x=4.5, color="grey", linewidth=1.5, linestyle="--", zorder=1, alpha=0.6)

# Horizontal line at y=0 removed per request

# ── Plot points (colored by regulation direction) ─────────────────────────
for grp in group_order:
    grp_data = data[data["label"] == grp]

    # Non-significant – grey
    nonsig = grp_data[~grp_data["is_sig"]]
    ax.scatter(
        nonsig["x"], nonsig["Log2FC"],
        c=NONSIG_COLOR, s=10, alpha=0.30, edgecolors="none", zorder=2,
    )

    # Significant up-regulated – red
    sig_up = grp_data[grp_data["is_sig"] & grp_data["is_up"]]
    ax.scatter(
        sig_up["x"], sig_up["Log2FC"],
        c=UP_COLOR, s=18, alpha=0.75, edgecolors="none", zorder=3,
    )

    # Significant down-regulated – blue
    sig_down = grp_data[grp_data["is_sig"] & grp_data["is_down"]]
    ax.scatter(
        sig_down["x"], sig_down["Log2FC"],
        c=DOWN_COLOR, s=18, alpha=0.75, edgecolors="none", zorder=3,
    )

# ── Center color bands with TIME-ONLY labels ──────────────────────────────
band_height = 0.75  # half-height of center label band
for i, grp in enumerate(group_order):
    color = BAND_COLORS[grp]
    time_label = TIME_LABELS[grp]

    rect = FancyBboxPatch(
        (i - 0.30, -band_height),
        0.60,
        2 * band_height,
        boxstyle="round,pad=0.04",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.92,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        i, 0, time_label,
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="white",
        zorder=6,
    )

# ── Axes styling ───────────────────────────────────────────────────────────
ax.set_xlim(-0.7, n_groups - 0.3)

y_abs_max = data["Log2FC"].abs().quantile(0.995)
y_abs_max = max(y_abs_max, 6)
ax.set_ylim(-y_abs_max * 1.18, y_abs_max * 1.18)

ax.set_ylabel("log₂FoldChange", fontsize=14, fontweight="bold")
ax.set_xlabel("")

ax.set_xticks(range(n_groups))
ax.set_xticklabels([""] * n_groups)

# Section headers
n_m = len(m_files)
n_ms = len(ms_files)
m_center = (n_m - 1) / 2
ms_center = n_m + (n_ms - 1) / 2

ax.text(
    m_center, -y_abs_max * 1.16,
    "Asymbiotic (M)",
    ha="center", va="top", fontsize=13, fontweight="bold", color="#555",
)
ax.text(
    ms_center, -y_abs_max * 1.16,
    "Symbiotic (MS)",
    ha="center", va="top", fontsize=13, fontweight="bold", color="#555",
)

# Up-regulated / Down-regulated side annotations
ax.text(
    -0.55, y_abs_max * 0.65,
    "Up-regulated ↑",
    ha="left", va="center", fontsize=11, fontweight="bold", color=UP_COLOR,
    rotation=90, zorder=8,
)
ax.text(
    -0.55, -y_abs_max * 0.65,
    "Down-regulated ↓",
    ha="left", va="center", fontsize=11, fontweight="bold", color=DOWN_COLOR,
    rotation=90, zorder=8,
)

# ── Legend ──────────────────────────────────────────────────────────────────
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=UP_COLOR,
           markersize=8, label='Up-regulated'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=DOWN_COLOR,
           markersize=8, label='Down-regulated'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=NONSIG_COLOR,
           markersize=8, label='Not Significant'),
]
ax.legend(handles=handles, loc="upper left", fontsize=10,
          framealpha=0.9, title="Regulation", title_fontsize=10,
          bbox_to_anchor=(1.02, 1.0), borderpad=1.0, handletextpad=1.2)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_bounds(-y_abs_max, y_abs_max)
ax.tick_params(bottom=False)

# Title
ax.set_title(
    "Combined Multi-group Differential Volcano Plot\n"
    "(Asymbiotic vs Symbiotic, Time: 10–50 d)",
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
