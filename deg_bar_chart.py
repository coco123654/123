#!/usr/bin/env python3
"""
Generate a multi-group differential gene expression (DEG) stacked bar chart.

Layout:
  - 5 time-point groups: Day-10, Day-20, Day-30, Day-40, Day-50
  - Each group has two bars: Asymbiotic (non-symbiotic) | Symbiotic
  - UP-regulated above zero, DOWN-regulated below zero
  - Bars are stacked: common genes (shared between Asym & Sym) + specific genes
  - Numbers annotated on each bar segment

Data source: M{time}VSM5.csv (Asymbiotic) and MS{time}VSM5.csv (Symbiotic)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_POINTS = [10, 20, 30, 40, 50]
TIME_LABELS = ["Day-10", "Day-20", "Day-30", "Day-40", "Day-50"]

# Colors matching the reference figure
COLOR_SPECIFIC_UP = "#FDEAAD"   # light yellow – specifically upregulated
COLOR_COMMON_UP = "#E8A317"     # orange/gold  – common upregulated
COLOR_COMMON_DOWN = "#4A7FB5"   # blue         – common downregulated
COLOR_SPECIFIC_DOWN = "#B0D4E8" # light blue   – specifically downregulated

# Layout parameters
FIGURE_SIZE = (16, 8)
BAR_WIDTH = 0.6
GROUP_GAP = 3.0       # distance between time-point groups
BAR_OFFSET = 0.12     # offset between Asym and Sym bars within a group

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ── Helper: read one DEG CSV and return sets of up/down Gene IDs ──────────
def read_deg_csv(filepath):
    """Return (set_of_up_genes, set_of_down_genes) from a DEG CSV file."""
    df = pd.read_csv(filepath)
    # Standardise column names (first 9 are always the same structure)
    cols = list(df.columns)
    cols[0] = "GeneID"
    cols[7] = "Significant"
    cols[8] = "Regulate"
    df.columns = cols

    df["Significant"] = df["Significant"].str.strip().str.lower()
    df["Regulate"] = df["Regulate"].str.strip().str.lower()

    sig = df[df["Significant"] == "yes"]
    up_genes = set(sig.loc[sig["Regulate"] == "up", "GeneID"])
    down_genes = set(sig.loc[sig["Regulate"] == "down", "GeneID"])
    return up_genes, down_genes


# ── Compute counts for every time-point ───────────────────────────────────
results = []

for tp in TIME_POINTS:
    asym_file = os.path.join(BASE_DIR, f"M{tp}VSM5.csv")
    sym_file = os.path.join(BASE_DIR, f"MS{tp}VSM5.csv")

    asym_up, asym_down = read_deg_csv(asym_file)
    sym_up, sym_down = read_deg_csv(sym_file)

    common_up = asym_up & sym_up
    common_down = asym_down & sym_down
    specific_up_asym = asym_up - sym_up
    specific_up_sym = sym_up - asym_up
    specific_down_asym = asym_down - sym_down
    specific_down_sym = sym_down - asym_down

    results.append(
        {
            "tp": tp,
            "common_up": len(common_up),
            "specific_up_asym": len(specific_up_asym),
            "specific_up_sym": len(specific_up_sym),
            "common_down": len(common_down),
            "specific_down_asym": len(specific_down_asym),
            "specific_down_sym": len(specific_down_sym),
        }
    )

    print(
        f"Day-{tp}: Asym UP={len(asym_up)} DOWN={len(asym_down)} | "
        f"Sym UP={len(sym_up)} DOWN={len(sym_down)} | "
        f"Common UP={len(common_up)} DOWN={len(common_down)}"
    )


# ── Draw the figure ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

n_tp = len(TIME_POINTS)

for idx, res in enumerate(results):
    x_center = idx * GROUP_GAP
    x_asym = x_center - BAR_WIDTH / 2 - BAR_OFFSET
    x_sym = x_center + BAR_WIDTH / 2 + BAR_OFFSET

    # ── UP bars (positive) ────────────────────────────────────────────────
    # Asymbiotic UP: common (bottom) + specific (top)
    ax.bar(x_asym, res["common_up"], BAR_WIDTH, color=COLOR_COMMON_UP,
           edgecolor="white", linewidth=0.5)
    ax.bar(x_asym, res["specific_up_asym"], BAR_WIDTH,
           bottom=res["common_up"], color=COLOR_SPECIFIC_UP,
           edgecolor="white", linewidth=0.5)

    # Symbiotic UP
    ax.bar(x_sym, res["common_up"], BAR_WIDTH, color=COLOR_COMMON_UP,
           edgecolor="white", linewidth=0.5)
    ax.bar(x_sym, res["specific_up_sym"], BAR_WIDTH,
           bottom=res["common_up"], color=COLOR_SPECIFIC_UP,
           edgecolor="white", linewidth=0.5)

    # ── DOWN bars (negative) ──────────────────────────────────────────────
    ax.bar(x_asym, -res["common_down"], BAR_WIDTH, color=COLOR_COMMON_DOWN,
           edgecolor="white", linewidth=0.5)
    ax.bar(x_asym, -res["specific_down_asym"], BAR_WIDTH,
           bottom=-res["common_down"], color=COLOR_SPECIFIC_DOWN,
           edgecolor="white", linewidth=0.5)

    ax.bar(x_sym, -res["common_down"], BAR_WIDTH, color=COLOR_COMMON_DOWN,
           edgecolor="white", linewidth=0.5)
    ax.bar(x_sym, -res["specific_down_sym"], BAR_WIDTH,
           bottom=-res["common_down"], color=COLOR_SPECIFIC_DOWN,
           edgecolor="white", linewidth=0.5)

    # ── Number annotations ────────────────────────────────────────────────
    def _label(x, y_mid, val, color="black", bold=False):
        if val > 0:
            ax.text(x, y_mid, str(val), ha="center", va="center",
                    fontsize=7, fontweight="bold" if bold else "normal",
                    color=color)

    # Common UP
    _label(x_asym, res["common_up"] / 2, res["common_up"], bold=True)
    _label(x_sym, res["common_up"] / 2, res["common_up"], bold=True)

    # Specific UP
    _label(x_asym, res["common_up"] + res["specific_up_asym"] / 2,
           res["specific_up_asym"])
    _label(x_sym, res["common_up"] + res["specific_up_sym"] / 2,
           res["specific_up_sym"])

    # Common DOWN
    _label(x_asym, -res["common_down"] / 2, res["common_down"],
           color="white", bold=True)
    _label(x_sym, -res["common_down"] / 2, res["common_down"],
           color="white", bold=True)

    # Specific DOWN
    _label(x_asym, -res["common_down"] - res["specific_down_asym"] / 2,
           res["specific_down_asym"])
    _label(x_sym, -res["common_down"] - res["specific_down_sym"] / 2,
           res["specific_down_sym"])

    # ── X-axis group labels (rotated) ─────────────────────────────────────
    ax.text(x_asym, 0, "Asymbiotic", ha="center", va="top",
            fontsize=8, rotation=45, transform=ax.get_xaxis_transform())
    ax.text(x_sym, 0, "Symbiotic", ha="center", va="top",
            fontsize=8, rotation=45, transform=ax.get_xaxis_transform())

# ── Time-point headers ────────────────────────────────────────────────────
for idx, label in enumerate(TIME_LABELS):
    x_center = idx * GROUP_GAP
    ax.text(x_center, 1.06, label, ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            transform=ax.get_xaxis_transform())

# ── Axes / labels ─────────────────────────────────────────────────────────
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xticks([])
ax.set_ylabel("No. of DEGs", fontsize=13, fontweight="bold")

# UP / DOWN side labels
ax.text(-0.04, 0.78, "UP", ha="right", va="center", fontsize=13,
        fontweight="bold", transform=ax.transAxes)
ax.text(-0.04, 0.22, "DOWN", ha="right", va="center", fontsize=13,
        fontweight="bold", transform=ax.transAxes)

# Legend
legend_elements = [
    Patch(facecolor=COLOR_SPECIFIC_UP, edgecolor="gray",
          label="Specifically upregulated genes"),
    Patch(facecolor=COLOR_COMMON_UP, edgecolor="gray",
          label="Common upregulated genes"),
    Patch(facecolor=COLOR_COMMON_DOWN, edgecolor="gray",
          label="Common downregulated genes"),
    Patch(facecolor=COLOR_SPECIFIC_DOWN, edgecolor="gray",
          label="Specifically downregulated genes"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
          frameon=True, edgecolor="gray")

# Clean up
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Transcriptome analysis of asymbiotically and symbiotically germinated fungus",
    fontsize=15, fontweight="bold", y=0.98,
)

# ── Save ──────────────────────────────────────────────────────────────────
out_png = os.path.join(BASE_DIR, "DEG_bar_chart_combined.png")
out_pdf = os.path.join(BASE_DIR, "DEG_bar_chart_combined.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\n✓ Saved: {out_png}")
print(f"✓ Saved: {out_pdf}")
