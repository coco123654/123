#!/usr/bin/env python3
"""
Time-series Clustering Heatmap of DEGs – Figure 3B style
=========================================================
Replicates the style of Figure 3B from the reference paper:
  - Left panel: Asymbiotic, Right panel: Symbiotic
  - Log2FC colour scale (blue–white–orange)
  - Hierarchical clustering with dendrogram on the left
  - Time points: 10, 20, 30, 40, 50 (days)
  - Genes grouped: symbiotic-upregulated together first,
    then non-symbiotic-downregulated together

Uses the union set of DEGs from both symbiotic (MS*VSM5) and
non-symbiotic (M*VSM5) comparisons.  The heatmap is drawn using
pheatmap-like style with a single dendrogram spanning all genes.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 200,
})

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD EXPRESSION DATA
# ═══════════════════════════════════════════════════════════════════════════

# Symbiotic TPM file (contains MS5, MS10..MS50 mean columns)
sym_tpm = pd.read_csv(
    "表达量分析结果表_Exp_G_RSEM_TPM_20230505_130348 (2).csv", index_col=0
)
# Non-symbiotic TPM file (contains M5, M10..M50 mean columns)
asym_tpm = pd.read_csv(
    "表达量分析结果表_Exp_G_RSEM_TPM_20230505_141523.csv", index_col=0
)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  COLLECT UNION DEGs
# ═══════════════════════════════════════════════════════════════════════════

time_points = [10, 20, 30, 40, 50]

sym_degs = set()
asym_degs = set()

# Collect per-time-point regulation direction for each gene
# key: gene_id, value: dict with 'sym_up', 'sym_down', 'asym_up', 'asym_down' counts
gene_regulation = {}

for tp in time_points:
    # Symbiotic DEGs (MS vs M5)
    ms_df = pd.read_csv(f"MS{tp}VSM5.csv")
    ms_sig = ms_df[ms_df["Significant"] == "yes"]
    sym_degs.update(ms_sig["Gene ID"].tolist())
    for _, row in ms_sig.iterrows():
        gid = row["Gene ID"]
        if gid not in gene_regulation:
            gene_regulation[gid] = {"sym_up": 0, "sym_down": 0,
                                    "asym_up": 0, "asym_down": 0}
        if row["Regulate"] == "up":
            gene_regulation[gid]["sym_up"] += 1
        else:
            gene_regulation[gid]["sym_down"] += 1

    # Non-symbiotic DEGs (M vs M5)
    m_df = pd.read_csv(f"M{tp}VSM5.csv")
    m_sig = m_df[m_df["Significant"] == "yes"]
    asym_degs.update(m_sig["Gene ID"].tolist())
    for _, row in m_sig.iterrows():
        gid = row["Gene ID"]
        if gid not in gene_regulation:
            gene_regulation[gid] = {"sym_up": 0, "sym_down": 0,
                                    "asym_up": 0, "asym_down": 0}
        if row["Regulate"] == "up":
            gene_regulation[gid]["asym_up"] += 1
        else:
            gene_regulation[gid]["asym_down"] += 1

all_degs = sym_degs | asym_degs
print(f"Symbiotic DEGs:     {len(sym_degs)}")
print(f"Non-symbiotic DEGs: {len(asym_degs)}")
print(f"Union DEGs:         {len(all_degs)}")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  BUILD LOG2FC MATRIX (vs M5 baseline)
# ═══════════════════════════════════════════════════════════════════════════

asym_mean_cols = [f"M{tp}" for tp in time_points]
sym_mean_cols  = [f"MS{tp}" for tp in time_points]

common_genes = sorted(all_degs & set(asym_tpm.index) & set(sym_tpm.index))
print(f"DEGs in both expression files: {len(common_genes)}")

pseudocount = 0.01

baseline_m5 = asym_tpm.loc[common_genes, "M5"].values.astype(float)

# Log2FC for non-symbiotic (M) vs M5
m_lfc = pd.DataFrame(index=common_genes)
for col in asym_mean_cols:
    vals = asym_tpm.loc[common_genes, col].values.astype(float)
    m_lfc[col] = np.log2((vals + pseudocount) / (baseline_m5 + pseudocount))

# Log2FC for symbiotic (MS) vs M5
ms_lfc = pd.DataFrame(index=common_genes)
for col in sym_mean_cols:
    vals = sym_tpm.loc[common_genes, col].values.astype(float)
    ms_lfc[col] = np.log2((vals + pseudocount) / (baseline_m5 + pseudocount))

m_lfc_arr  = m_lfc.values    # shape (n, 5): M10..M50
ms_lfc_arr = ms_lfc.values   # shape (n, 5): MS10..MS50

# Combined matrix: [Asymbiotic | Symbiotic]
lfc_combined = np.hstack([m_lfc_arr, ms_lfc_arr])

# ═══════════════════════════════════════════════════════════════════════════
# 4.  CLASSIFY GENES INTO TWO MAJOR GROUPS
#     Group A: Symbiotic upregulated (sym mean LFC > 0)
#     Group B: Non-symbiotic downregulated / rest
#     Within each group, use hierarchical clustering
# ═══════════════════════════════════════════════════════════════════════════

ms_mean_lfc = np.mean(ms_lfc_arr, axis=1)   # mean log2FC across symbiotic time points
m_mean_lfc  = np.mean(m_lfc_arr, axis=1)    # mean log2FC across asymbiotic time points

# Classify into groups for ordering:
# Priority 1: Symbiotic UP & Asymbiotic DOWN  (orange in sym, blue in asym)
# Priority 2: Symbiotic UP & Asymbiotic UP    (orange in both)
# Priority 3: Symbiotic DOWN & Asymbiotic DOWN (blue in both)
# Priority 4: Symbiotic DOWN & Asymbiotic UP   (blue in sym, orange in asym)
groups = np.zeros(len(common_genes), dtype=int)
for i in range(len(common_genes)):
    sym_up = ms_mean_lfc[i] > 0
    asym_up = m_mean_lfc[i] > 0
    if sym_up and not asym_up:
        groups[i] = 0   # sym↑ asym↓
    elif sym_up and asym_up:
        groups[i] = 1   # both up
    elif not sym_up and not asym_up:
        groups[i] = 2   # both down
    else:
        groups[i] = 3   # sym↓ asym↑

print(f"\nGene groups:")
group_names = {0: "Sym UP / Asym DOWN", 1: "Both UP",
               2: "Both DOWN", 3: "Sym DOWN / Asym UP"}
for g in range(4):
    print(f"  Group {g} ({group_names[g]}): {np.sum(groups == g)} genes")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  ORDER ROWS: group by regulation direction, cluster within each group
# ═══════════════════════════════════════════════════════════════════════════

def cluster_and_order(indices, data):
    """Hierarchically cluster rows and return ordered indices."""
    if len(indices) <= 2:
        return indices.tolist()
    Z = linkage(data[indices], method="ward", metric="euclidean")
    order = leaves_list(Z)
    return indices[order].tolist()


ordered_idx = []
group_boundaries = []  # (start, end, group_id)

for g in [0, 1, 2, 3]:
    mask = np.where(groups == g)[0]
    if len(mask) == 0:
        continue
    start = len(ordered_idx)
    clustered = cluster_and_order(mask, lfc_combined)
    ordered_idx.extend(clustered)
    group_boundaries.append((start, len(ordered_idx), g))

ordered_idx = np.array(ordered_idx)
heatmap_matrix = lfc_combined[ordered_idx]
n_genes = heatmap_matrix.shape[0]

print(f"\nTotal genes in heatmap: {n_genes}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  COMPUTE FULL DENDROGRAM for display
# ═══════════════════════════════════════════════════════════════════════════

# Compute a single linkage for the entire ordered matrix for the dendrogram
if n_genes > 2:
    Z_full = linkage(heatmap_matrix, method="ward", metric="euclidean")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  PLOT (Figure 3B pheatmap style)
# ═══════════════════════════════════════════════════════════════════════════

fig_h = 10
fig_w = 6

fig = plt.figure(figsize=(fig_w, fig_h))

# GridSpec: [dendrogram | asym_heatmap | gap | sym_heatmap]
gs = gridspec.GridSpec(
    2, 4,
    width_ratios=[1.2, 4, 0.15, 4],
    height_ratios=[0.5, 20],
    wspace=0.02, hspace=0.03,
)

# ── Colour scale: blue – white – orange (matching reference paper) ──
cmap = LinearSegmentedColormap.from_list(
    "bwo",
    ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0",
     "#F7F7F7",
     "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"],
    N=256,
)

vmax_val = min(15, np.percentile(np.abs(heatmap_matrix), 99))
vmin_val = -vmax_val

# ── Top row: condition labels ──
ax_top0 = fig.add_subplot(gs[0, 0])
ax_top0.axis("off")

ax_asym_label = fig.add_subplot(gs[0, 1])
ax_asym_label.set_xlim(0, 1)
ax_asym_label.set_ylim(0, 1)
ax_asym_label.text(
    0.5, 0.3, "Asymbiotic", ha="center", va="center",
    fontsize=11, fontweight="bold", color="#333333",
)
ax_asym_label.axis("off")

ax_gap_top = fig.add_subplot(gs[0, 2])
ax_gap_top.axis("off")

ax_sym_label = fig.add_subplot(gs[0, 3])
ax_sym_label.set_xlim(0, 1)
ax_sym_label.set_ylim(0, 1)
ax_sym_label.text(
    0.5, 0.3, "Symbiotic", ha="center", va="center",
    fontsize=11, fontweight="bold", color="#333333",
)
ax_sym_label.axis("off")

# ── Dendrogram (left) ──
ax_dendro = fig.add_subplot(gs[1, 0])

if n_genes > 2:
    d = dendrogram(
        Z_full,
        orientation="left",
        ax=ax_dendro,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#333333",
    )

    ax_dendro.set_xlim(ax_dendro.get_xlim()[::-1])  # flip x so root is on right
ax_dendro.axis("off")

# ── Gap column ──
ax_gap = fig.add_subplot(gs[1, 2])
ax_gap.axis("off")

# ── Heatmap: Asymbiotic (left panel) ──
ax_asym = fig.add_subplot(gs[1, 1])
im_asym = ax_asym.imshow(
    heatmap_matrix[:, :5],
    aspect="auto",
    cmap=cmap,
    vmin=vmin_val,
    vmax=vmax_val,
    interpolation="nearest",
)
ax_asym.set_xticks(range(5))
ax_asym.set_xticklabels([str(tp) for tp in time_points], fontsize=9)
ax_asym.xaxis.set_ticks_position("bottom")
ax_asym.tick_params(axis="x", length=3, pad=2)
ax_asym.set_yticks([])

# ── Heatmap: Symbiotic (right panel) ──
ax_sym = fig.add_subplot(gs[1, 3])
im_sym = ax_sym.imshow(
    heatmap_matrix[:, 5:],
    aspect="auto",
    cmap=cmap,
    vmin=vmin_val,
    vmax=vmax_val,
    interpolation="nearest",
)
ax_sym.set_xticks(range(5))
ax_sym.set_xticklabels([str(tp) for tp in time_points], fontsize=9)
ax_sym.xaxis.set_ticks_position("bottom")
ax_sym.tick_params(axis="x", length=3, pad=2)
ax_sym.set_yticks([])

# ── Shared x-axis label ──
fig.text(0.55, 0.02, "(Day)", ha="center", fontsize=10)

# ── Colour bar (right side, vertical) ──
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
cbar = fig.colorbar(im_sym, cax=cbar_ax, orientation="vertical")
cbar.ax.tick_params(labelsize=8, length=2, pad=2)
cbar.set_ticks([vmin_val, 0, vmax_val])
cbar.set_ticklabels([f"{vmin_val:.0f}", "0", f"{vmax_val:.0f}"])
cbar_ax.set_ylabel("Log$_2$FC", fontsize=9, rotation=270, labelpad=12)

# ── Save ──
plt.savefig(
    "timeseries_clustering_heatmap.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.savefig(
    "timeseries_clustering_heatmap.pdf",
    bbox_inches="tight",
    facecolor="white",
)
print("\nSaved: timeseries_clustering_heatmap.png / .pdf")
plt.close()
