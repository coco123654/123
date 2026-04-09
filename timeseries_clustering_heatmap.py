#!/usr/bin/env python3
"""
Time-series Clustering Heatmap of DEGs (Symbiotic vs Non-symbiotic)
===================================================================
Replicates the style of Figure 3B from the reference paper:
  - Left: Asymbiotic (non-symbiotic), Right: Symbiotic
  - Log2FC colour scale (blue–white–orange)
  - Hierarchical clustering with dendrogram
  - Row-side colour annotation for 4 temporal pattern categories
  - Time points labelled as 10, 20, 30, 40, 50 (days)
  - Within EACH category, genes where symbiotic is up-regulated AND
    non-symbiotic is down-regulated are grouped together first.

Categories:
  1. 共生早期快速诱导型       – Early rapid induction in symbiosis
  2. 共生中期持续升高型       – Mid-term continuous increase in symbiosis
  3. 共生后期稳定维持型       – Late stable maintenance in symbiosis
  4. 非共生衰减而共生保持激活型 – Asymbiotic decay, symbiotic maintained
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
import warnings

warnings.filterwarnings("ignore")

# ── Font configuration (CJK support) ──────────────────────────────────────
_cjk_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_cjk_font_path)
    _cjk_prop = fm.FontProperties(fname=_cjk_font_path)
    _cjk_family = _cjk_prop.get_name()
except Exception:
    _cjk_family = "sans-serif"

plt.rcParams.update({
    "font.family": _cjk_family,
    "axes.unicode_minus": False,
    "figure.dpi": 200,
})

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD EXPRESSION DATA
# ═══════════════════════════════════════════════════════════════════════════

asym_tpm = pd.read_csv(
    "表达量分析结果表_Exp_G_RSEM_TPM_20230505_141523.csv", index_col=0
)
sym_tpm = pd.read_csv(
    "表达量分析结果表_Exp_G_RSEM_TPM_20230505_130348 (2).csv", index_col=0
)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  COLLECT UNION DEGs (共生 vs 非共生 差异基因)
# ═══════════════════════════════════════════════════════════════════════════

time_points = [10, 20, 30, 40, 50]

sym_degs = set()
asym_degs = set()

for tp in time_points:
    # Symbiotic DEGs (MS vs M5)
    ms_df = pd.read_csv(f"MS{tp}VSM5.csv")
    ms_sig = ms_df[ms_df["Significant"] == "yes"]
    sym_degs.update(ms_sig["Gene ID"].tolist())

    # Non-symbiotic DEGs (M vs M5)
    m_df = pd.read_csv(f"M{tp}VSM5.csv")
    m_sig = m_df[m_df["Significant"] == "yes"]
    asym_degs.update(m_sig["Gene ID"].tolist())

all_degs = sym_degs | asym_degs
print(f"Symbiotic DEGs: {len(sym_degs)}")
print(f"Non-symbiotic DEGs: {len(asym_degs)}")
print(f"Union DEGs: {len(all_degs)}")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  BUILD LOG2FC MATRIX (vs M5 baseline)
# ═══════════════════════════════════════════════════════════════════════════

asym_mean_cols = [f"M{tp}" for tp in time_points]
sym_mean_cols  = [f"MS{tp}" for tp in time_points]

common_genes = sorted(all_degs & set(asym_tpm.index) & set(sym_tpm.index))
print(f"DEGs present in both expression files: {len(common_genes)}")

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

m_lfc_arr  = m_lfc.values
ms_lfc_arr = ms_lfc.values

lfc_combined = np.hstack([m_lfc_arr, ms_lfc_arr])

# ═══════════════════════════════════════════════════════════════════════════
# 4.  CLASSIFY GENES INTO 4 TEMPORAL PATTERN CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════

labels = []
# For ALL categories: determine sub-group (sym-up & asym-down first)
sub_groups = []  # 0 = sym-up & asym-down, 1 = sym-up & asym-up, 2 = sym-down & asym-down, 3 = others

for i, gene in enumerate(common_genes):
    m  = m_lfc_arr[i]
    ms = ms_lfc_arr[i]

    ms_early = ms[0]
    ms_mid   = np.mean(ms[1:3])
    ms_late  = np.mean(ms[3:5])

    m_early = m[0]
    m_mid   = np.mean(m[1:3])
    m_late  = np.mean(m[3:5])

    ms_mean = np.mean(ms)
    m_mean  = np.mean(m)

    scores = {}

    # Category 1: 共生早期快速诱导型
    scores[1] = (
        1.5 * (ms_early - ms_mid)
        + (ms_early - ms_late)
        + 1.5 * (ms_early - m_early)
    )

    # Category 2: 共生中期持续升高型
    scores[2] = (
        1.5 * (ms_mid - ms_early)
        + (ms_mid - m_mid)
        + 0.5 * (ms_mean - m_mean)
    )

    # Category 3: 共生后期稳定维持型
    scores[3] = (
        (ms_late - ms_early)
        + 1.5 * (ms_late - m_late)
        + 0.5 * (ms_late - ms_mid)
    )

    # Category 4: 非共生衰减而共生保持激活型
    scores[4] = (
        1.5 * (m_early - m_late)
        + (ms_mean - m_mean)
        + 0.5 * (ms_late - m_late)
    )

    best = max(scores, key=scores.get)
    labels.append(best)

    # Determine sub-group for sorting within each category using
    # overall mean log2FC direction (threshold = 0).
    # This groups genes with opposite regulation patterns together.
    is_sym_up = ms_mean > 0
    is_sym_down = ms_mean < 0
    is_asym_up = m_mean > 0
    is_asym_down = m_mean < 0

    if is_sym_up and is_asym_down:
        sub_groups.append(0)   # sym↑ asym↓ (排最前)
    elif is_sym_up and is_asym_up:
        sub_groups.append(1)   # both up
    elif is_sym_down and is_asym_down:
        sub_groups.append(2)   # both down
    elif is_sym_down and is_asym_up:
        sub_groups.append(3)   # sym↓ asym↑
    else:
        sub_groups.append(4)   # other

labels = np.array(labels)
sub_groups = np.array(sub_groups)

category_names = {
    1: "共生早期快速诱导型",
    2: "共生中期持续升高型",
    3: "共生后期稳定维持型",
    4: "非共生衰减而共生保持激活型",
}

category_colors = {
    1: "#E8443A",
    2: "#F5A623",
    3: "#4CAF50",
    4: "#2979B9",
}

for cat_id in sorted(category_names.keys()):
    n = np.sum(labels == cat_id)
    # Count sub-groups
    cat_mask = labels == cat_id
    n_sym_up_asym_down = np.sum((cat_mask) & (sub_groups == 0))
    print(f"  Category {cat_id} ({category_names[cat_id]}): {n} genes "
          f"(sym↑asym↓: {n_sym_up_asym_down})")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  ORDER ROWS: group by category, sub-sort within each, cluster within
# ═══════════════════════════════════════════════════════════════════════════

def cluster_indices(indices, data):
    """Hierarchically cluster a set of row indices and return ordered indices."""
    if len(indices) <= 2:
        return indices.tolist()
    sub_data = data[indices]
    Z = linkage(sub_data, method="ward", metric="euclidean")
    order = leaves_list(Z)
    return indices[order].tolist()


ordered_idx = []
cat_boundaries = []
# Track sub-group boundaries within each category for optional annotation
subgroup_info = []  # (start, end, cat_id, sub_group_id)

row_cursor = 0
for cat_id in sorted(category_names.keys()):
    mask = np.where(labels == cat_id)[0]
    if len(mask) == 0:
        continue

    cat_start = row_cursor

    # Sub-sort: group 0 (sym-up & asym-down) first, then 1, 2, 3, 4
    for sg in [0, 1, 2, 3, 4]:
        sg_mask = mask[sub_groups[mask] == sg]
        if len(sg_mask) == 0:
            continue
        sg_start = row_cursor
        clustered = cluster_indices(sg_mask, lfc_combined)
        ordered_idx.extend(clustered)
        row_cursor += len(clustered)
        subgroup_info.append((sg_start, row_cursor, cat_id, sg))

    cat_boundaries.append((cat_start, row_cursor, cat_id))

ordered_idx = np.array(ordered_idx)
heatmap_matrix = lfc_combined[ordered_idx]
ordered_labels = labels[ordered_idx]
ordered_sub_groups = sub_groups[ordered_idx]

n_genes = heatmap_matrix.shape[0]
print(f"\nTotal genes in heatmap: {n_genes}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  COMPUTE DENDROGRAMS PER CATEGORY (for display)
# ═══════════════════════════════════════════════════════════════════════════

cat_linkages = {}
for start, end, cat_id in cat_boundaries:
    n_cat = end - start
    if n_cat > 2:
        Z = linkage(heatmap_matrix[start:end], method="ward", metric="euclidean")
        cat_linkages[cat_id] = (Z, start, n_cat)

# ═══════════════════════════════════════════════════════════════════════════
# 7.  PLOT (Figure 3B pheatmap style)
# ═══════════════════════════════════════════════════════════════════════════

fig_h = 14  # compact height matching pheatmap style
fig_w = 8

fig = plt.figure(figsize=(fig_w, fig_h))

# GridSpec layout:
#   Row 0: [empty | empty | colorbar_mini | asym_label | gap | sym_label | empty]
#   Row 1: [dendro | row_color | gap | asym_heat | gap | sym_heat | empty]
#   Row 2: [empty | empty | empty | legend_area spanning cols | empty]
gs = gridspec.GridSpec(
    3, 7,
    width_ratios=[0.8, 0.2, 0.04, 3.5, 0.06, 3.5, 0.04],
    height_ratios=[0.4, 20, 0.8],
    wspace=0.015, hspace=0.03,
)

# ── Colour scale (blue–white–orange, matching the reference paper) ──
cmap = LinearSegmentedColormap.from_list(
    "ref_bwo",
    ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"],
    N=256,
)

vmax_val = min(15, np.percentile(np.abs(heatmap_matrix), 99))
vmin_val = -vmax_val

# ── Top row: condition labels & colorbar ──
for ci in [0, 1, 2, 6]:
    ax_ = fig.add_subplot(gs[0, ci])
    ax_.axis("off")

ax_asym_label = fig.add_subplot(gs[0, 3])
ax_asym_label.set_xlim(0, 1); ax_asym_label.set_ylim(0, 1)
ax_asym_label.axhspan(0.15, 0.85, color="#E8943A", alpha=0.9)
ax_asym_label.text(
    0.5, 0.5, "Asymbiotic", ha="center", va="center",
    fontsize=11, fontweight="bold", color="white",
)
ax_asym_label.axis("off")

ax_gap_top = fig.add_subplot(gs[0, 4])
ax_gap_top.axis("off")

ax_sym_label = fig.add_subplot(gs[0, 5])
ax_sym_label.set_xlim(0, 1); ax_sym_label.set_ylim(0, 1)
ax_sym_label.axhspan(0.15, 0.85, color="#1B6FB5", alpha=0.9)
ax_sym_label.text(
    0.5, 0.5, "Symbiotic", ha="center", va="center",
    fontsize=11, fontweight="bold", color="white",
)
ax_sym_label.axis("off")

# ── Dendrogram (left): per-category sub-dendrograms ──
ax_dendro = fig.add_subplot(gs[1, 0])
ax_dendro.set_ylim(0, n_genes)
ax_dendro.invert_yaxis()

for cat_id, (Z_cat, row_start, n_cat) in cat_linkages.items():
    fig_tmp, ax_tmp = plt.subplots(figsize=(2, 2))
    d = dendrogram(
        Z_cat,
        orientation="left",
        ax=ax_tmp,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#333333",
    )
    plt.close(fig_tmp)

    icoord = np.array(d["icoord"])
    dcoord = np.array(d["dcoord"])

    y_min_orig = 5.0
    y_max_orig = 5.0 + (n_cat - 1) * 10.0
    y_range_orig = y_max_orig - y_min_orig if y_max_orig != y_min_orig else 1.0

    y_min_target = row_start + 0.5
    y_max_target = row_start + n_cat - 0.5

    x_max = dcoord.max() if dcoord.max() > 0 else 1.0

    for ic, dc in zip(icoord, dcoord):
        yy = y_min_target + (np.array(ic) - y_min_orig) / y_range_orig * (
            y_max_target - y_min_target
        )
        xx = 1.0 - np.array(dc) / x_max
        ax_dendro.plot(xx, yy, color="#333333", lw=0.25, solid_capstyle="round")

ax_dendro.set_xlim(0, 1)
ax_dendro.axis("off")

# ── Row-side colour bar (category annotation) ──
ax_row = fig.add_subplot(gs[1, 1])
row_colors = np.array([category_colors[l] for l in ordered_labels])
for i, c in enumerate(row_colors):
    ax_row.axhspan(i, i + 1, color=c, lw=0)
ax_row.set_ylim(0, n_genes)
ax_row.set_xlim(0, 1)
ax_row.invert_yaxis()
for start, end, _ in cat_boundaries:
    ax_row.axhline(y=start, color="white", lw=1.2)
    ax_row.axhline(y=end, color="white", lw=1.2)
ax_row.axis("off")

# ── Gap columns ──
ax_gap1 = fig.add_subplot(gs[1, 2])
ax_gap1.axis("off")
ax_gap2 = fig.add_subplot(gs[1, 4])
ax_gap2.axis("off")
ax_gap3 = fig.add_subplot(gs[1, 6])
ax_gap3.axis("off")

# ── Heatmap: Asymbiotic (left) ──
ax_asym = fig.add_subplot(gs[1, 3])
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
for start, end, _ in cat_boundaries:
    ax_asym.axhline(y=start - 0.5, color="white", lw=1.8)
    ax_asym.axhline(y=end - 0.5, color="white", lw=1.8)
ax_asym.set_xlabel("(d)", fontsize=9, labelpad=3)

# ── Heatmap: Symbiotic (right) ──
ax_sym = fig.add_subplot(gs[1, 5])
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
for start, end, _ in cat_boundaries:
    ax_sym.axhline(y=start - 0.5, color="white", lw=1.8)
    ax_sym.axhline(y=end - 0.5, color="white", lw=1.8)
ax_sym.set_xlabel("(d)", fontsize=9, labelpad=3)

# ── Colour bar (inset in top-right area of symbiotic panel) ──
# Place a small horizontal colorbar above the symbiotic heatmap
cbar_ax = fig.add_axes([0.72, 0.91, 0.15, 0.012])  # [left, bottom, width, height]
cbar = fig.colorbar(im_sym, cax=cbar_ax, orientation="horizontal")
cbar.ax.tick_params(labelsize=7, length=2, pad=1)
cbar.set_ticks([vmin_val, 0, vmax_val])
cbar.set_ticklabels([f"{vmin_val:.0f}", "0", f"{vmax_val:.0f}"])
cbar_ax.set_title("Log$_2$FC", fontsize=8, pad=3)

# ── Category legend (bottom row) ──
ax_legend = fig.add_subplot(gs[2, :])
ax_legend.axis("off")
legend_patches = [
    Patch(facecolor=category_colors[k], label=f"{k}. {category_names[k]}")
    for k in sorted(category_names.keys())
]
legend = ax_legend.legend(
    handles=legend_patches,
    loc="center",
    ncol=2,
    fontsize=8,
    frameon=True,
    edgecolor="#CCCCCC",
    fancybox=False,
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=1.5,
)

# ── Title ──
fig.suptitle(
    "Hierarchical Clustering of DEGs – Asymbiotic vs Symbiotic (10–50 d)",
    fontsize=12,
    fontweight="bold",
    y=0.97,
)

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
