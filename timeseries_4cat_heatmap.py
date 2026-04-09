#!/usr/bin/env python3
"""
Time-series Clustering Heatmap – 4 Gene Categories
====================================================
Finds differential genes between symbiotic (MS) and non-symbiotic (M) conditions,
classifies them into 4 temporal pattern categories, and generates a heatmap with
row-side colour annotations matching the reference figure style.

Categories:
  1. 共生早期快速诱导型   – Early rapid induction in symbiosis
  2. 共生中期持续升高型   – Mid-term continuous increase in symbiosis
  3. 共生后期稳定维持型   – Late stable maintenance in symbiosis
  4. 非共生衰减而共生保持激活型 – Non-symbiotic decay, symbiotic maintained

Layout:
  Left panel  : Symbiotic  (with green→blue gradient top bar)
  Right panel : Non-symbiotic (with green→yellow gradient top bar)
  Row-side colour bar on the left for gene cluster membership
  Horizontal gaps between clusters
  Z-score colour scale: blue (-2) – white (0) – orange (+2)
  X-axis labels: 10, 20, 30, 40, 50
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
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

sym_tpm = pd.read_csv(
    "表达量分析结果表_Exp_G_RSEM_TPM_20230505_130348 (2).csv", index_col=0
)
asym_tpm = pd.read_csv(
    "表达量分析结果表_Exp_G_RSEM_TPM_20230505_141523.csv", index_col=0
)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  COLLECT UNION DEGs
# ═══════════════════════════════════════════════════════════════════════════

time_points = [10, 20, 30, 40, 50]
all_degs = set()

for tp in time_points:
    for prefix in ["M", "MS"]:
        fname = f"{prefix}{tp}VSM5.csv"
        df = pd.read_csv(fname)
        sig = df[df["Significant"] == "yes"]
        all_degs.update(sig["Gene ID"].tolist())

print(f"Union DEGs: {len(all_degs)}")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  BUILD EXPRESSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════

asym_cols = [f"M{tp}" for tp in time_points]
sym_cols  = [f"MS{tp}" for tp in time_points]

common_genes = sorted(all_degs & set(asym_tpm.index) & set(sym_tpm.index))
print(f"DEGs in both expression files: {len(common_genes)}")

# Build expression matrix: columns = [M10..M50, MS10..MS50]
expr = pd.DataFrame(index=common_genes)
for col in asym_cols:
    expr[col] = asym_tpm.loc[common_genes, col].values.astype(float)
for col in sym_cols:
    expr[col] = sym_tpm.loc[common_genes, col].values.astype(float)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Z-SCORE NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

row_mean = expr.mean(axis=1)
row_std  = expr.std(axis=1).replace(0, 1)
zscore = expr.sub(row_mean, axis=0).div(row_std, axis=0)
zscore = zscore.clip(-3, 3)

# ═══════════════════════════════════════════════════════════════════════════
# 5.  COMPUTE log2FC FOR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

pseudocount = 0.01
baseline_m5 = asym_tpm.loc[common_genes, "M5"].values.astype(float)

m_lfc = pd.DataFrame(index=common_genes)
for col in asym_cols:
    vals = asym_tpm.loc[common_genes, col].values.astype(float)
    m_lfc[col] = np.log2((vals + pseudocount) / (baseline_m5 + pseudocount))

ms_lfc = pd.DataFrame(index=common_genes)
for col in sym_cols:
    vals = sym_tpm.loc[common_genes, col].values.astype(float)
    ms_lfc[col] = np.log2((vals + pseudocount) / (baseline_m5 + pseudocount))

m_lfc_arr  = m_lfc.values   # shape (n, 5): M10..M50
ms_lfc_arr = ms_lfc.values  # shape (n, 5): MS10..MS50

# ═══════════════════════════════════════════════════════════════════════════
# 6.  CLASSIFY GENES INTO 4 TEMPORAL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

labels = []

for i in range(len(common_genes)):
    m  = m_lfc_arr[i]    # M10..M50 log2FC vs M5
    ms = ms_lfc_arr[i]   # MS10..MS50 log2FC vs M5

    # Symbiotic early, mid, late phases
    ms_early = ms[0]                    # MS10
    ms_mid   = np.mean(ms[1:3])         # mean(MS20, MS30)
    ms_late  = np.mean(ms[3:5])         # mean(MS40, MS50)

    # Non-symbiotic early, mid, late phases
    m_early = m[0]                      # M10
    m_mid   = np.mean(m[1:3])           # mean(M20, M30)
    m_late  = np.mean(m[3:5])           # mean(M40, M50)

    ms_mean = np.mean(ms)
    m_mean  = np.mean(m)

    scores = {}

    # Category 1: 共生早期快速诱导型
    # Symbiotic peaks early; MS10 >> MS_mid/late and MS10 >> M10
    scores[1] = ((ms_early - ms_mid) + (ms_early - ms_late)
                 + 1.5 * (ms_early - m_early))

    # Category 2: 共生中期持续升高型
    # Symbiotic rises from early→mid, mid clearly higher than asymbiotic
    scores[2] = (1.5 * (ms_mid - ms_early) + (ms_mid - m_mid)
                 + 0.5 * (ms_mean - m_mean))

    # Category 3: 共生后期稳定维持型
    # Symbiotic high in late stage, higher than asymbiotic late
    scores[3] = ((ms_late - ms_early) + 1.5 * (ms_late - m_late)
                 + 0.5 * (ms_late - ms_mid))

    # Category 4: 非共生衰减而共生保持激活型
    # Asymbiotic decays (early > late), symbiotic stays active
    scores[4] = (1.5 * (m_early - m_late) + (ms_mean - m_mean)
                 + 0.5 * (ms_late - m_late))

    labels.append(max(scores, key=scores.get))

labels = np.array(labels)

category_names = {
    1: "共生早期快速诱导型",
    2: "共生中期持续升高型",
    3: "共生后期稳定维持型",
    4: "非共生衰减而共生保持激活型",
}

category_colors = {
    1: "#D94F27",   # red-orange
    2: "#E8943A",   # orange
    3: "#3670B5",   # blue
    4: "#4FAE82",   # teal-green
}

for cat_id in sorted(category_names.keys()):
    n = np.sum(labels == cat_id)
    print(f"  Category {cat_id} ({category_names[cat_id]}): {n} genes")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  ORDER ROWS: group by category, cluster within each group
# ═══════════════════════════════════════════════════════════════════════════

ordered_idx = []
group_boundaries = []  # (start, end, cat_id)

for cat_id in sorted(category_names.keys()):
    mask = np.where(labels == cat_id)[0]
    if len(mask) == 0:
        continue
    start = len(ordered_idx)
    if len(mask) <= 2:
        ordered_idx.extend(mask.tolist())
    else:
        sub = zscore.values[mask]
        Z = linkage(sub, method="ward", metric="euclidean")
        order = leaves_list(Z)
        ordered_idx.extend(mask[order].tolist())
    group_boundaries.append((start, len(ordered_idx), cat_id))

ordered_idx = np.array(ordered_idx)
heatmap_matrix = zscore.values[ordered_idx]
ordered_labels = labels[ordered_idx]
n_genes = heatmap_matrix.shape[0]

print(f"\nTotal genes in heatmap: {n_genes}")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  PLOT (reference figure style)
# ═══════════════════════════════════════════════════════════════════════════

# Insert small gaps between categories for visual separation
gap_size = max(2, int(n_genes * 0.008))
n_gaps = len(group_boundaries) - 1
total_rows = n_genes + n_gaps * gap_size

# Build gapped heatmap matrix and colour-bar array
gapped_matrix = np.full((total_rows, 10), np.nan)
gapped_colors = np.full(total_rows, -1, dtype=int)

row_ptr = 0
category_midpoints = {}  # cat_id -> mid row for labelling

for idx, (start, end, cat_id) in enumerate(group_boundaries):
    n_rows = end - start
    gapped_matrix[row_ptr:row_ptr + n_rows] = heatmap_matrix[start:end]
    gapped_colors[row_ptr:row_ptr + n_rows] = cat_id
    category_midpoints[cat_id] = row_ptr + n_rows // 2
    row_ptr += n_rows
    if idx < len(group_boundaries) - 1:
        row_ptr += gap_size  # leave gap

# ── Figure layout ──
fig_h = max(10, min(20, total_rows * 0.015))
fig_w = 8.5

fig = plt.figure(figsize=(fig_w, fig_h))

# GridSpec: row0=top bar, row1=heatmap
# Columns: [row-colour-bar | sym heatmap | gap | asym heatmap | colorbar]
gs = gridspec.GridSpec(
    2, 5,
    width_ratios=[0.4, 4.5, 0.15, 4.5, 0.6],
    height_ratios=[0.5, 20],
    wspace=0.02, hspace=0.04,
)

# ── Colour palette: blue–white–orange ──
cmap = LinearSegmentedColormap.from_list(
    "bwo",
    ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0",
     "#F7F7F7",
     "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"],
    N=256,
)
cmap.set_bad(color="white")  # NaN gaps → white

vmin_val, vmax_val = -2, 2

# ── Top row: condition gradient bars ──
ax_top_left = fig.add_subplot(gs[0, 0])
ax_top_left.axis("off")

# Symbiotic top bar (green → blue gradient)
ax_sym_top = fig.add_subplot(gs[0, 1])
sym_grad = np.linspace(0, 1, 256).reshape(1, -1)
sym_cmap = LinearSegmentedColormap.from_list("sym_grad", ["#7CBB6E", "#2C7BB6"], N=256)
ax_sym_top.imshow(sym_grad, aspect="auto", cmap=sym_cmap, extent=[0, 5, 0, 1])
ax_sym_top.set_xlim(0, 5)
ax_sym_top.axis("off")
ax_sym_top.set_title("Symbiotic", fontsize=11, fontweight="bold", pad=4)

ax_gap_top = fig.add_subplot(gs[0, 2])
ax_gap_top.axis("off")

# Non-symbiotic top bar (green → yellow gradient)
ax_asym_top = fig.add_subplot(gs[0, 3])
asym_grad = np.linspace(0, 1, 256).reshape(1, -1)
asym_cmap = LinearSegmentedColormap.from_list("asym_grad", ["#7CBB6E", "#E8D44D"], N=256)
ax_asym_top.imshow(asym_grad, aspect="auto", cmap=asym_cmap, extent=[0, 5, 0, 1])
ax_asym_top.set_xlim(0, 5)
ax_asym_top.axis("off")
ax_asym_top.set_title("Non-symbiotic", fontsize=11, fontweight="bold", pad=4)

ax_top_right = fig.add_subplot(gs[0, 4])
ax_top_right.axis("off")

# ── Row-side colour bar ──
ax_row = fig.add_subplot(gs[1, 0])
for i in range(total_rows):
    cat_id = gapped_colors[i]
    if cat_id > 0:
        ax_row.axhspan(i, i + 1, color=category_colors[cat_id], lw=0)
    else:
        ax_row.axhspan(i, i + 1, color="white", lw=0)

ax_row.set_ylim(0, total_rows)
ax_row.set_xlim(0, 1)
ax_row.invert_yaxis()
ax_row.axis("off")

# ── Heatmap: Symbiotic (LEFT panel, columns MS10..MS50 = indices 5..9) ──
ax_sym = fig.add_subplot(gs[1, 1])
im_sym = ax_sym.imshow(
    gapped_matrix[:, 5:],
    aspect="auto", cmap=cmap, vmin=vmin_val, vmax=vmax_val,
    interpolation="nearest",
)
ax_sym.set_xticks(range(5))
ax_sym.set_xticklabels([str(tp) for tp in time_points], fontsize=10, fontweight="bold")
ax_sym.xaxis.set_ticks_position("bottom")
ax_sym.tick_params(axis="x", length=3, pad=3)
ax_sym.set_yticks([])

# ── Gap column ──
ax_gap = fig.add_subplot(gs[1, 2])
ax_gap.axis("off")

# ── Heatmap: Non-symbiotic (RIGHT panel, columns M10..M50 = indices 0..4) ──
ax_asym = fig.add_subplot(gs[1, 3])
im_asym = ax_asym.imshow(
    gapped_matrix[:, :5],
    aspect="auto", cmap=cmap, vmin=vmin_val, vmax=vmax_val,
    interpolation="nearest",
)
ax_asym.set_xticks(range(5))
ax_asym.set_xticklabels([str(tp) for tp in time_points], fontsize=10, fontweight="bold")
ax_asym.xaxis.set_ticks_position("bottom")
ax_asym.tick_params(axis="x", length=3, pad=3)
ax_asym.set_yticks([])

# ── Colour bar (right side) ──
cbar_ax = fig.add_axes([0.91, 0.25, 0.015, 0.4])
cbar = fig.colorbar(im_sym, cax=cbar_ax, orientation="vertical")
cbar.set_ticks([-2, 0, 2])
cbar.set_ticklabels(["-2", "0", "2"])
cbar.ax.tick_params(labelsize=9, length=2, pad=2)

# ── Legend for categories ──
legend_handles = [
    Patch(facecolor=category_colors[cat_id], label=category_names[cat_id])
    for cat_id in sorted(category_names.keys())
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    fontsize=8.5,
    frameon=True,
    edgecolor="#CCCCCC",
    bbox_to_anchor=(0.5, -0.02),
    handlelength=1.5,
    handleheight=1.0,
)

# ── Save ──
plt.savefig(
    "timeseries_4cat_heatmap.png",
    dpi=200, bbox_inches="tight", facecolor="white",
)
plt.savefig(
    "timeseries_4cat_heatmap.pdf",
    bbox_inches="tight", facecolor="white",
)
print("\nSaved: timeseries_4cat_heatmap.png / .pdf")
plt.close()
