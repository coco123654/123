#!/usr/bin/env python3
"""
Union DEG Time-course Atlas (10-50 d)
Classify differentially expressed genes between symbiotic (MS) and
non-symbiotic (M) conditions into temporal pattern categories,
and generate a heatmap with row-side colour annotations.

Categories:
  1. 共生早期快速诱导型   – Early rapid induction in symbiosis
  2. 共生中期持续升高型   – Mid-term continuous increase in symbiosis
  3. 共生后期稳定维持型   – Late stable maintenance in symbiosis
  4. 非共生衰减而共生保持激活型 – Asymbiotic decay, symbiotic maintained
  补充亚型A: 双条件持续抑制型  – Suppressed under both conditions
  补充亚型B: 共生恢复波动型    – Symbiotic recovery / fluctuation
  补充亚型C: 非共生偏高残留型  – Higher residual in asymbiotic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list
import warnings

warnings.filterwarnings("ignore")

# ── Matplotlib font configuration ──────────────────────────────────────────
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
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

asym_tpm = pd.read_csv(
    "表达量分析结果表_Exp_G_RSEM_TPM_20230505_141523.csv", index_col=0
)
sym_tpm = pd.read_csv(
    "表达量分析结果表_Exp_G_RSEM_TPM_20230505_130348 (2).csv", index_col=0
)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  COLLECT UNION DEGs FROM EXISTING CSV FILES
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
# 3.  BUILD EXPRESSION MATRIX  (mean TPM)
# ═══════════════════════════════════════════════════════════════════════════

# Non-symbiotic mean columns: M10 M20 M30 M40 M50
# Symbiotic  mean columns: MS10 MS20 MS30 MS40 MS50
asym_cols = [f"M{tp}" for tp in time_points]   # columns in asym_tpm
sym_cols  = [f"MS{tp}" for tp in time_points]   # columns in sym_tpm

# Keep only genes present in both expression files
common_genes = sorted(all_degs & set(asym_tpm.index) & set(sym_tpm.index))
print(f"DEGs in both expression files: {len(common_genes)}")

# Build matrix: rows = genes, columns = M10..M50, MS10..MS50
expr = pd.DataFrame(index=common_genes)
for col in asym_cols:
    expr[col] = asym_tpm.loc[common_genes, col].values.astype(float)
for col in sym_cols:
    expr[col] = sym_tpm.loc[common_genes, col].values.astype(float)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Z-SCORE NORMALISATION (for heatmap visualisation)
# ═══════════════════════════════════════════════════════════════════════════

row_mean = expr.mean(axis=1)
row_std  = expr.std(axis=1).replace(0, 1)  # avoid div-by-zero
zscore = expr.sub(row_mean, axis=0).div(row_std, axis=0)
zscore = zscore.clip(-3, 3)

# ═══════════════════════════════════════════════════════════════════════════
# 5.  COMPUTE log2FC VS BASELINE (M5) FOR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

pseudocount = 0.01

# Baseline M5 from the non-symbiotic file (shared baseline for both conditions)
baseline_m5 = asym_tpm.loc[common_genes, "M5"].values.astype(float)

# log2FC for asymbiotic (M) vs M5 baseline
m_lfc = pd.DataFrame(index=common_genes)
for col in asym_cols:
    vals = asym_tpm.loc[common_genes, col].values.astype(float)
    m_lfc[col] = np.log2((vals + pseudocount) / (baseline_m5 + pseudocount))

# log2FC for symbiotic (MS) vs M5 baseline
ms_lfc = pd.DataFrame(index=common_genes)
for col in sym_cols:
    vals = sym_tpm.loc[common_genes, col].values.astype(float)
    ms_lfc[col] = np.log2((vals + pseudocount) / (baseline_m5 + pseudocount))

m_lfc_arr  = m_lfc.values   # shape (n, 5)  M10..M50
ms_lfc_arr = ms_lfc.values  # shape (n, 5)  MS10..MS50

# ═══════════════════════════════════════════════════════════════════════════
# 6.  CLASSIFY GENES INTO TEMPORAL PATTERNS (using log2FC)
# ═══════════════════════════════════════════════════════════════════════════

labels = []  # category label per gene

for i in range(len(common_genes)):
    m  = m_lfc_arr[i]    # M10..M50  log2FC vs M5
    ms = ms_lfc_arr[i]   # MS10..MS50 log2FC vs M5

    # Symbiotic early, mid, late
    ms_early = ms[0]
    ms_mid   = np.mean(ms[1:3])
    ms_late  = np.mean(ms[3:5])

    # Asymbiotic early, mid, late
    m_early = m[0]
    m_mid   = np.mean(m[1:3])
    m_late  = np.mean(m[3:5])

    ms_mean = np.mean(ms)
    m_mean  = np.mean(m)

    scores = {}

    # Category 1: 共生早期快速诱导型
    # Symbiotic peaks early; MS10 >> MS_mid/late; MS10 >> M10
    scores[1] = (ms_early - ms_mid) + (ms_early - ms_late) + 1.5 * (ms_early - m_early)

    # Category 2: 共生中期持续升高型
    # Symbiotic rises from early to mid, mid clearly higher than asymbiotic
    scores[2] = 1.5 * (ms_mid - ms_early) + (ms_mid - m_mid) + 0.5 * (ms_mean - m_mean)

    # Category 3: 共生后期稳定维持型
    # Symbiotic high in late, higher than asymbiotic late
    scores[3] = (ms_late - ms_early) + 1.5 * (ms_late - m_late) + 0.5 * (ms_late - ms_mid)

    # Category 4: 非共生衰减而共生保持激活型
    # Asymbiotic decays (early > late), symbiotic stays active
    scores[4] = 1.5 * (m_early - m_late) + (ms_mean - m_mean) + 0.5 * (ms_late - m_late)

    # Category 5: 补充亚型A: 双条件持续抑制型
    # Both conditions downregulated vs baseline; both M and MS < 0
    scores[5] = -1.2 * ms_mean - 1.2 * m_mean + 0.5 * min(ms_mean, m_mean)

    # Category 6: 补充亚型B: 共生恢复波动型
    # Symbiotic shows non-monotonic fluctuation
    ms_range = np.max(ms) - np.min(ms)
    ms_nonmono = abs(ms[2] - ms[0]) + abs(ms[-1] - ms[2])
    scores[6] = ms_nonmono + 0.5 * ms_range + 0.8 * (ms_mean - m_mean) - abs(ms[-1] - ms[0])

    # Category 7: 补充亚型C: 非共生偏高残留型
    # Asymbiotic generally higher than symbiotic
    scores[7] = 1.5 * (m_mean - ms_mean) + 0.5 * (m_late - ms_late)

    labels.append(max(scores, key=scores.get))

labels = np.array(labels)

category_names = {
    1: "共生早期快速诱导型",
    2: "共生中期持续升高型",
    3: "共生后期稳定维持型",
    4: "非共生衰减而共生保\n持激活型",
    5: "补充亚型A:\n双条件持续抑制型",
    6: "补充亚型B:\n共生恢复波动型",
    7: "补充亚型C:\n非共生偏高残留型",
}

category_colors = {
    1: "#D94F27",   # red-orange
    2: "#E8943A",   # orange
    3: "#4FAE82",   # teal-green
    4: "#3670B5",   # blue
    5: "#8B5CA0",   # purple
    6: "#D45B9A",   # pink
    7: "#666666",   # grey
}

for cat_id in sorted(category_names.keys()):
    n = np.sum(labels == cat_id)
    print(f"  Category {cat_id} ({category_names[cat_id].replace(chr(10),' ')}): {n} genes")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  ORDER ROWS: group by category, cluster within each group
# ═══════════════════════════════════════════════════════════════════════════

ordered_idx = []
for cat_id in sorted(category_names.keys()):
    mask = np.where(labels == cat_id)[0]
    if len(mask) <= 2:
        ordered_idx.extend(mask.tolist())
        continue
    sub = zscore.values[mask]
    dist = np.clip(sub, -3, 3)
    Z = linkage(dist, method="ward", metric="euclidean")
    order = leaves_list(Z)
    ordered_idx.extend(mask[order].tolist())

ordered_idx = np.array(ordered_idx)
heatmap_matrix = zscore.values[ordered_idx]
ordered_labels = labels[ordered_idx]

# ═══════════════════════════════════════════════════════════════════════════
# 7.  PLOT
# ═══════════════════════════════════════════════════════════════════════════

n_genes = heatmap_matrix.shape[0]
fig_h = max(14, min(40, n_genes * 0.005))

fig = plt.figure(figsize=(10, fig_h))
gs = gridspec.GridSpec(
    2, 3,
    width_ratios=[0.6, 5, 5],
    height_ratios=[0.4, 20],
    wspace=0.02, hspace=0.02,
)

# --- colour-bar legend (top) ---
ax_asym_bar = fig.add_subplot(gs[0, 1])
ax_sym_bar  = fig.add_subplot(gs[0, 2])

ax_asym_bar.axhspan(0, 1, color="#D96430", lw=0)
ax_asym_bar.set_xlim(0, 1); ax_asym_bar.set_ylim(0, 1)
ax_asym_bar.set_title("Asymbiotic", fontsize=12, fontweight="bold", color="#D96430")
ax_asym_bar.axis("off")

ax_sym_bar.axhspan(0, 1, color="#1B6FB5", lw=0)
ax_sym_bar.set_xlim(0, 1); ax_sym_bar.set_ylim(0, 1)
ax_sym_bar.set_title("Symbiotic", fontsize=12, fontweight="bold", color="#1B6FB5")
ax_sym_bar.axis("off")

# Hide top-left cell
ax_topleft = fig.add_subplot(gs[0, 0])
ax_topleft.axis("off")

# --- row side colour bar ---
ax_row = fig.add_subplot(gs[1, 0])
row_colors = np.array([category_colors[l] for l in ordered_labels])

# Draw row-side colour strip
for i, c in enumerate(row_colors):
    ax_row.axhspan(i, i + 1, color=c, lw=0)

ax_row.set_ylim(0, n_genes)
ax_row.set_xlim(0, 1)
ax_row.invert_yaxis()
ax_row.axis("off")

# Add category labels on the left side of the colour bar
prev_cat = None
seg_start = 0
for i, cat in enumerate(ordered_labels):
    if cat != prev_cat and prev_cat is not None:
        mid = (seg_start + i) / 2
        ax_row.text(
            -0.1, mid, category_names[prev_cat],
            ha="right", va="center", fontsize=7.5,
            fontweight="bold", color=category_colors[prev_cat],
        )
        seg_start = i
    prev_cat = cat
# last segment
mid = (seg_start + n_genes) / 2
ax_row.text(
    -0.1, mid, category_names[prev_cat],
    ha="right", va="center", fontsize=7.5,
    fontweight="bold", color=category_colors[prev_cat],
)

# --- heatmap: asymbiotic half ---
ax_asym = fig.add_subplot(gs[1, 1])
cmap = LinearSegmentedColormap.from_list(
    "custom_bwr", ["#2166AC", "#D1E5F0", "#FEFEBE", "#FDBF6F", "#D94F27"], N=256
)
im_asym = ax_asym.imshow(
    heatmap_matrix[:, :5],
    aspect="auto", cmap=cmap, vmin=-2, vmax=2,
    interpolation="nearest",
)
ax_asym.set_xticks(range(5))
ax_asym.set_xticklabels([f"M{tp}" for tp in time_points], fontsize=9)
ax_asym.xaxis.set_ticks_position("bottom")
ax_asym.set_yticks([])

# --- heatmap: symbiotic half ---
ax_sym = fig.add_subplot(gs[1, 2])
im_sym = ax_sym.imshow(
    heatmap_matrix[:, 5:],
    aspect="auto", cmap=cmap, vmin=-2, vmax=2,
    interpolation="nearest",
)
ax_sym.set_xticks(range(5))
ax_sym.set_xticklabels([f"MS{tp}" for tp in time_points], fontsize=9)
ax_sym.xaxis.set_ticks_position("bottom")
ax_sym.set_yticks([])

# --- colour bar ---
cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.35])
cbar = fig.colorbar(im_sym, cax=cbar_ax)
cbar.set_ticks([-2, -1, 0, 1, 2])
cbar.ax.tick_params(labelsize=8)

# --- overall title ---
fig.suptitle("Union DEG Time-course Atlas (10-50 d)", fontsize=14, fontweight="bold", y=0.98)

plt.savefig("union_deg_timecourse_atlas.png", dpi=200, bbox_inches="tight")
plt.savefig("union_deg_timecourse_atlas.pdf", bbox_inches="tight")
print("\nSaved: union_deg_timecourse_atlas.png / .pdf")
plt.close()
