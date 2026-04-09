#!/usr/bin/env python3
"""
Time-series Clustering Heatmap of DEGs (Symbiotic vs Non-symbiotic)
===================================================================
Replicates the style of Figure 3B from the reference paper:
  - Left: Asymbiotic (non-symbiotic), Right: Symbiotic
  - Log2FC colour scale (blue–white–orange)
  - Hierarchical clustering with dendrogram
  - Row-side colour annotation for 4 temporal pattern categories
  - Time points labelled as 10, 20, 30, 40, 50

Categories:
  1. 共生早期快速诱导型       – Early rapid induction in symbiosis
  2. 共生中期持续升高型       – Mid-term continuous increase in symbiosis
  3. 共生后期稳定维持型       – Late stable maintenance in symbiosis
  4. 非共生衰减而共生保持激活型 – Asymbiotic decay, symbiotic maintained
     Sub-sorted: symbiotic-up & non-symbiotic-down genes grouped first
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
fm.fontManager.addfont(_cjk_font_path)
_cjk_prop = fm.FontProperties(fname=_cjk_font_path)
_cjk_family = _cjk_prop.get_name()

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

# Collect DEGs that are differentially expressed in symbiotic OR non-symbiotic
# compared to the M5 baseline
sym_degs = set()     # DEGs in symbiotic condition
asym_degs = set()    # DEGs in non-symbiotic condition

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

# Mean TPM columns
asym_mean_cols = [f"M{tp}" for tp in time_points]    # M10 M20 M30 M40 M50
sym_mean_cols  = [f"MS{tp}" for tp in time_points]    # MS10 MS20 MS30 MS40 MS50

# Keep genes present in both expression files
common_genes = sorted(all_degs & set(asym_tpm.index) & set(sym_tpm.index))
print(f"DEGs present in both expression files: {len(common_genes)}")

pseudocount = 0.01

# Baseline M5 (from the non-symbiotic file; both files share the same M5)
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

m_lfc_arr  = m_lfc.values    # shape (n_genes, 5)
ms_lfc_arr = ms_lfc.values   # shape (n_genes, 5)

# Combined matrix: [M10..M50, MS10..MS50] all in log2FC
lfc_combined = np.hstack([m_lfc_arr, ms_lfc_arr])  # (n_genes, 10)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  CLASSIFY GENES INTO 4 TEMPORAL PATTERN CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════

labels = []       # category label per gene
sub_sort = []     # sub-sort key for category 4

for i in range(len(common_genes)):
    m  = m_lfc_arr[i]     # M10..M50 log2FC vs M5
    ms = ms_lfc_arr[i]    # MS10..MS50 log2FC vs M5

    # Phase summaries for symbiotic
    ms_early = ms[0]                 # 10d
    ms_mid   = np.mean(ms[1:3])      # 20-30d
    ms_late  = np.mean(ms[3:5])      # 40-50d

    # Phase summaries for non-symbiotic
    m_early = m[0]
    m_mid   = np.mean(m[1:3])
    m_late  = np.mean(m[3:5])

    ms_mean = np.mean(ms)
    m_mean  = np.mean(m)

    # --- Scoring for each category ---
    scores = {}

    # Category 1: 共生早期快速诱导型
    # Symbiotic peaks early (MS10 high, then drops or stays flat); MS10 > M10
    scores[1] = (
        1.5 * (ms_early - ms_mid)
        + (ms_early - ms_late)
        + 1.5 * (ms_early - m_early)
    )

    # Category 2: 共生中期持续升高型
    # Symbiotic rises from early to mid; mid clearly > early; mid > asymbiotic mid
    scores[2] = (
        1.5 * (ms_mid - ms_early)
        + (ms_mid - m_mid)
        + 0.5 * (ms_mean - m_mean)
    )

    # Category 3: 共生后期稳定维持型
    # Symbiotic high in late phase; late > early; late > asymbiotic late
    scores[3] = (
        (ms_late - ms_early)
        + 1.5 * (ms_late - m_late)
        + 0.5 * (ms_late - ms_mid)
    )

    # Category 4: 非共生衰减而共生保持激活型
    # Non-symbiotic decays (early -> late drops); symbiotic stays active
    scores[4] = (
        1.5 * (m_early - m_late)
        + (ms_mean - m_mean)
        + 0.5 * (ms_late - m_late)
    )

    best = max(scores, key=scores.get)
    labels.append(best)

    # Sub-sort for category 4: prioritize sym-up & asym-down
    # Positive = sym up AND asym down (should come first)
    is_sym_up = ms_mean > 0.5
    is_asym_down = m_late < -0.5
    sub_sort.append(1 if (is_sym_up and is_asym_down) else 0)

labels = np.array(labels)
sub_sort = np.array(sub_sort)

category_names = {
    1: "共生早期快速诱导型",
    2: "共生中期持续升高型",
    3: "共生后期稳定维持型",
    4: "非共生衰减而共生保持激活型",
}

category_colors = {
    1: "#E8443A",   # red
    2: "#F5A623",   # orange
    3: "#4CAF50",   # green
    4: "#2979B9",   # blue
}

for cat_id in sorted(category_names.keys()):
    n = np.sum(labels == cat_id)
    print(f"  Category {cat_id} ({category_names[cat_id]}): {n} genes")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  ORDER ROWS: group by category, sub-sort cat4, cluster within groups
# ═══════════════════════════════════════════════════════════════════════════

ordered_idx = []
cat_linkages = {}   # {cat_id: (Z, n_genes_before, n_genes_in_cat)}
cat_boundaries = [] # [(start_row, end_row, cat_id), ...]

row_cursor = 0
for cat_id in sorted(category_names.keys()):
    mask = np.where(labels == cat_id)[0]
    if len(mask) == 0:
        continue

    if cat_id == 4:
        # Within category 4: first group sym-up & asym-down, then others
        group_a = mask[sub_sort[mask] == 1]  # sym-up & asym-down
        group_b = mask[sub_sort[mask] == 0]  # others
        combined_cat4 = []
        for sub_mask in [group_a, group_b]:
            if len(sub_mask) <= 2:
                combined_cat4.extend(sub_mask.tolist())
            else:
                sub_data = lfc_combined[sub_mask]
                Z = linkage(sub_data, method="ward", metric="euclidean")
                order = leaves_list(Z)
                combined_cat4.extend(sub_mask[order].tolist())
        ordered_idx.extend(combined_cat4)
        n_cat = len(combined_cat4)
        # Compute linkage for the whole category 4 (for dendrogram display)
        if n_cat > 2:
            cat_linkages[cat_id] = (
                linkage(lfc_combined[combined_cat4], method="ward"),
                row_cursor,
                n_cat,
            )
    else:
        if len(mask) <= 2:
            ordered_idx.extend(mask.tolist())
            n_cat = len(mask)
        else:
            sub_data = lfc_combined[mask]
            Z = linkage(sub_data, method="ward", metric="euclidean")
            order = leaves_list(Z)
            ordered_idx.extend(mask[order].tolist())
            n_cat = len(mask)
            cat_linkages[cat_id] = (Z, row_cursor, n_cat)

    cat_boundaries.append((row_cursor, row_cursor + n_cat, cat_id))
    row_cursor += n_cat

ordered_idx = np.array(ordered_idx)
heatmap_matrix = lfc_combined[ordered_idx]
ordered_labels = labels[ordered_idx]

# ═══════════════════════════════════════════════════════════════════════════
# 6.  PLOT (Figure B style)
# ═══════════════════════════════════════════════════════════════════════════

n_genes = heatmap_matrix.shape[0]
fig_h = max(10, min(18, n_genes * 0.004 + 4))

fig = plt.figure(figsize=(10, fig_h))

# GridSpec: [dendrogram | row_color | asym_heatmap | sym_heatmap | colorbar_space]
gs = gridspec.GridSpec(
    2, 5,
    width_ratios=[1.2, 0.3, 4, 4, 0.15],
    height_ratios=[0.3, 20],
    wspace=0.02, hspace=0.03,
)

# ── Colour scale (blue–white–orange, matching the reference) ──
cmap = LinearSegmentedColormap.from_list(
    "ref_bwo",
    ["#2166AC", "#92C5DE", "#F7F7F7", "#FDBF6F", "#E8601C"],
    N=256,
)

# Determine vmin/vmax based on data range (clip extreme outliers)
vmax_val = min(15, np.percentile(np.abs(heatmap_matrix), 99))
vmin_val = -vmax_val

# ── Top labels (Asymbiotic / Symbiotic) ──
ax_top_dendro = fig.add_subplot(gs[0, 0])
ax_top_dendro.axis("off")
ax_top_color = fig.add_subplot(gs[0, 1])
ax_top_color.axis("off")

ax_asym_label = fig.add_subplot(gs[0, 2])
ax_asym_label.set_xlim(0, 1)
ax_asym_label.set_ylim(0, 1)
ax_asym_label.axhspan(0.3, 0.7, color="#E8943A", alpha=0.8)
ax_asym_label.text(
    0.5, 0.5, "Asymbiotic", ha="center", va="center",
    fontsize=13, fontweight="bold", color="white",
)
ax_asym_label.axis("off")

ax_sym_label = fig.add_subplot(gs[0, 3])
ax_sym_label.set_xlim(0, 1)
ax_sym_label.set_ylim(0, 1)
ax_sym_label.axhspan(0.3, 0.7, color="#1B6FB5", alpha=0.8)
ax_sym_label.text(
    0.5, 0.5, "Symbiotic", ha="center", va="center",
    fontsize=13, fontweight="bold", color="white",
)
ax_sym_label.axis("off")

ax_top_cb = fig.add_subplot(gs[0, 4])
ax_top_cb.axis("off")

# ── Dendrogram (left): per-category sub-dendrograms ──
ax_dendro = fig.add_subplot(gs[1, 0])
ax_dendro.set_ylim(0, n_genes)
ax_dendro.invert_yaxis()

for cat_id, (Z_cat, row_start, n_cat) in cat_linkages.items():
    # Render dendrogram into a temporary axes to extract coordinates
    fig_tmp, ax_tmp = plt.subplots(figsize=(2, 2))
    d = dendrogram(
        Z_cat,
        orientation="left",
        ax=ax_tmp,
        no_labels=True,
        color_threshold=0,
        above_threshold_color=category_colors[cat_id],
    )
    plt.close(fig_tmp)

    # Map dendrogram coordinates into the main axes
    icoord = np.array(d["icoord"])  # y coords (leaf positions)
    dcoord = np.array(d["dcoord"])  # x coords (distances)

    # Original leaf range: 5 to 5 + (n_cat-1)*10
    y_min_orig = 5.0
    y_max_orig = 5.0 + (n_cat - 1) * 10.0
    y_range_orig = y_max_orig - y_min_orig if y_max_orig != y_min_orig else 1.0

    # Target range in row-space
    y_min_target = row_start + 0.5
    y_max_target = row_start + n_cat - 0.5

    # Normalise x (distance) to [0, 1]
    x_max = dcoord.max() if dcoord.max() > 0 else 1.0

    for ic, dc in zip(icoord, dcoord):
        # Map y (icoord) from original leaf space to target row space
        yy = y_min_target + (np.array(ic) - y_min_orig) / y_range_orig * (
            y_max_target - y_min_target
        )
        # Map x (dcoord) — flip so root is at left edge
        xx = 1.0 - np.array(dc) / x_max
        ax_dendro.plot(xx, yy, color=category_colors[cat_id], lw=0.4, solid_capstyle="round")

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

# Draw horizontal lines between categories
for start, end, cat_id in cat_boundaries:
    ax_row.axhline(y=start, color="white", lw=1.0)
    ax_row.axhline(y=end, color="white", lw=1.0)
ax_row.axis("off")

# ── Heatmap: Asymbiotic (left half) ──
ax_asym = fig.add_subplot(gs[1, 2])
im_asym = ax_asym.imshow(
    heatmap_matrix[:, :5],
    aspect="auto",
    cmap=cmap,
    vmin=vmin_val,
    vmax=vmax_val,
    interpolation="nearest",
)
ax_asym.set_xticks(range(5))
ax_asym.set_xticklabels([f"{tp}" for tp in time_points], fontsize=10)
ax_asym.xaxis.set_ticks_position("bottom")
ax_asym.set_yticks([])
# Draw category boundary lines
for start, end, cat_id in cat_boundaries:
    ax_asym.axhline(y=start - 0.5, color="white", lw=1.5)
    ax_asym.axhline(y=end - 0.5, color="white", lw=1.5)

# ── Heatmap: Symbiotic (right half) ──
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
ax_sym.set_xticklabels([f"{tp}" for tp in time_points], fontsize=10)
ax_sym.xaxis.set_ticks_position("bottom")
ax_sym.set_yticks([])
# Draw category boundary lines
for start, end, cat_id in cat_boundaries:
    ax_sym.axhline(y=start - 0.5, color="white", lw=1.5)
    ax_sym.axhline(y=end - 0.5, color="white", lw=1.5)

ax_sym.set_xlabel("(d)", fontsize=10)
ax_asym.set_xlabel("(d)", fontsize=10)

# ── Colour bar (right edge) ──
cbar_ax = fig.add_subplot(gs[1, 4])
cbar = fig.colorbar(im_sym, cax=cbar_ax)
cbar.set_label("Log$_2$FC", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# ── Category legend (bottom) ──
legend_patches = [
    Patch(facecolor=category_colors[k], label=f"{k}. {category_names[k]}")
    for k in sorted(category_names.keys())
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=2,
    fontsize=9,
    frameon=True,
    edgecolor="#CCCCCC",
    bbox_to_anchor=(0.55, -0.02),
)

# ── Title ──
fig.suptitle(
    "Hierarchical Clustering of DEGs – Asymbiotic vs Symbiotic (10–50 d)",
    fontsize=14,
    fontweight="bold",
    y=0.99,
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
