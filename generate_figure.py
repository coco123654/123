#!/usr/bin/env python3
"""
Generate Figure 3-style transcriptome analysis figure.
Panels:
  A) Bar chart of DEG counts (upregulated/downregulated, common/specific)
  B) Heatmap with hierarchical clustering of log2FC
  C) GO enrichment dot plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import fisher_exact
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Configure matplotlib for Chinese text ──────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

# --- Asymbiotic TPM ---
asym_tpm = pd.read_csv(
    '表达量分析结果表_Exp_G_RSEM_TPM_20230505_141523.csv',
    index_col=0
)

# --- Symbiotic TPM ---
sym_tpm = pd.read_csv(
    '表达量分析结果表_Exp_G_RSEM_TPM_20230505_130348 (2).csv',
    index_col=0
)

# --- Symbiotic count file (for GO annotations) ---
sym_count = pd.read_csv(
    'gene.count.matrix.annot.xls',
    sep='\t',
    index_col=0
)

# Extract GO annotations from symbiotic count file
go_annotations = sym_count['go'].fillna('').to_dict()

# Also try to get GO annotations from TPM file
go_from_tpm_asym = asym_tpm['go_term'].fillna('').to_dict() if 'go_term' in asym_tpm.columns else {}
go_from_tpm_sym = sym_tpm['go_term'].fillna('').to_dict() if 'go_term' in sym_tpm.columns else {}

# Merge GO annotations (use any available source)
for gene_id in go_annotations:
    if not go_annotations[gene_id]:
        if gene_id in go_from_tpm_asym and go_from_tpm_asym[gene_id]:
            go_annotations[gene_id] = str(go_from_tpm_asym[gene_id])
        elif gene_id in go_from_tpm_sym and go_from_tpm_sym[gene_id]:
            go_annotations[gene_id] = str(go_from_tpm_sym[gene_id])

for gene_id in go_from_tpm_asym:
    if gene_id not in go_annotations or not go_annotations[gene_id]:
        go_annotations[gene_id] = str(go_from_tpm_asym[gene_id])
for gene_id in go_from_tpm_sym:
    if gene_id not in go_annotations or not go_annotations[gene_id]:
        go_annotations[gene_id] = str(go_from_tpm_sym[gene_id])

# ═══════════════════════════════════════════════════════════════════════════
# 2.  DEG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

# Mean TPM columns
# Asymbiotic: M5 (baseline), M10, M20, M30, M40, M50
# Symbiotic:  M5 (baseline), MS5, MS10, MS20, MS30, MS40, MS50

# Time points for comparison (paired)
time_points = [10, 20, 30, 40, 50]
time_labels = ['Day-10', 'Day-20', 'Day-30', 'Day-40', 'Day-50']

pseudocount = 0.01  # small pseudocount to avoid log(0)
fc_threshold = 1.0  # |log2FC| >= 1

# Get common gene set
asym_genes = set(asym_tpm.index)
sym_genes = set(sym_tpm.index)
common_genes = sorted(asym_genes & sym_genes)
print(f"Common genes between asymbiotic and symbiotic: {len(common_genes)}")

# Calculate log2FC for each gene at each time point
# Asymbiotic: log2(M_time / M5)
# Symbiotic:  log2(MS_time / M5)

asym_log2fc = {}
sym_log2fc = {}

for tp in time_points:
    asym_col_mean = f'M{tp}'
    sym_col_mean = f'MS{tp}'
    baseline_col = 'M5'

    asym_fc_vals = {}
    sym_fc_vals = {}

    for gene in common_genes:
        # Asymbiotic fold change
        baseline_val = float(asym_tpm.loc[gene, baseline_col]) if gene in asym_tpm.index else 0
        asym_val = float(asym_tpm.loc[gene, asym_col_mean]) if gene in asym_tpm.index else 0
        asym_fc_vals[gene] = np.log2((asym_val + pseudocount) / (baseline_val + pseudocount))

        # Symbiotic fold change
        baseline_val_sym = float(sym_tpm.loc[gene, baseline_col]) if gene in sym_tpm.index else 0
        sym_val = float(sym_tpm.loc[gene, sym_col_mean]) if gene in sym_tpm.index else 0
        sym_fc_vals[gene] = np.log2((sym_val + pseudocount) / (baseline_val_sym + pseudocount))

    asym_log2fc[tp] = asym_fc_vals
    sym_log2fc[tp] = sym_fc_vals

# Classify DEGs at each time point
deg_results = {}

for tp in time_points:
    asym_up = set()
    asym_down = set()
    sym_up = set()
    sym_down = set()

    for gene in common_genes:
        afc = asym_log2fc[tp][gene]
        sfc = sym_log2fc[tp][gene]

        if afc >= fc_threshold:
            asym_up.add(gene)
        elif afc <= -fc_threshold:
            asym_down.add(gene)

        if sfc >= fc_threshold:
            sym_up.add(gene)
        elif sfc <= -fc_threshold:
            sym_down.add(gene)

    common_up = asym_up & sym_up
    common_down = asym_down & sym_down
    specific_up_asym = asym_up - sym_up
    specific_up_sym = sym_up - asym_up
    specific_down_asym = asym_down - sym_down
    specific_down_sym = sym_down - asym_down

    deg_results[tp] = {
        'common_up': len(common_up),
        'specific_up_asym': len(specific_up_asym),
        'specific_up_sym': len(specific_up_sym),
        'common_down': len(common_down),
        'specific_down_asym': len(specific_down_asym),
        'specific_down_sym': len(specific_down_sym),
        'common_up_genes': common_up,
        'common_down_genes': common_down,
        'all_asym_up': asym_up,
        'all_asym_down': asym_down,
        'all_sym_up': sym_up,
        'all_sym_down': sym_down,
    }

    print(f"\nDay {tp}:")
    print(f"  Asymbiotic UP: {len(asym_up)} (common: {len(common_up)}, specific: {len(specific_up_asym)})")
    print(f"  Symbiotic  UP: {len(sym_up)} (common: {len(common_up)}, specific: {len(specific_up_sym)})")
    print(f"  Asymbiotic DOWN: {len(asym_down)} (common: {len(common_down)}, specific: {len(specific_down_asym)})")
    print(f"  Symbiotic  DOWN: {len(sym_down)} (common: {len(common_down)}, specific: {len(specific_down_sym)})")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  COLLECT ALL DEGs FOR HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

all_degs = set()
for tp in time_points:
    res = deg_results[tp]
    all_degs |= res['all_asym_up'] | res['all_asym_down']
    all_degs |= res['all_sym_up'] | res['all_sym_down']

print(f"\nTotal unique DEGs across all time points: {len(all_degs)}")

# Build heatmap matrix: rows=genes, columns=Asym_10..50, Sym_10..50
heatmap_cols_asym = [f'Asym_{tp}d' for tp in time_points]
heatmap_cols_sym = [f'Sym_{tp}d' for tp in time_points]
heatmap_cols = heatmap_cols_asym + heatmap_cols_sym

heatmap_data = np.zeros((len(all_degs), len(heatmap_cols)))
all_degs_list = sorted(all_degs)

for i, gene in enumerate(all_degs_list):
    for j, tp in enumerate(time_points):
        heatmap_data[i, j] = asym_log2fc[tp].get(gene, 0)
        heatmap_data[i, j + len(time_points)] = sym_log2fc[tp].get(gene, 0)

# Clip extreme values for better visualization
heatmap_data = np.clip(heatmap_data, -15, 15)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  GO ENRICHMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def parse_go_terms(go_str):
    """Parse GO term annotations from various formats."""
    terms = []
    if not go_str or go_str == '------' or pd.isna(go_str):
        return terms

    go_str = str(go_str)

    # Format from count file: "GO:0016787(molecular_function:hydrolase activity)"
    # Format from TPM file: "GO:0016787" with "MF:hydrolase activity;"
    import re
    # Pattern 1: GO:XXXXXXX(category:description)
    pattern1 = re.findall(r'(GO:\d+)\(([^:]+):([^)]+)\)', go_str)
    for go_id, category, desc in pattern1:
        cat_map = {
            'molecular_function': 'MF',
            'biological_process': 'BP',
            'cellular_component': 'CC',
        }
        cat = cat_map.get(category.strip(), category.strip())
        terms.append((go_id, cat, desc.strip()))

    # Pattern 2: "MF:hydrolase activity;" or "BP:..." or "CC:..."
    if not pattern1:
        pattern2 = re.findall(r'(BP|MF|CC):([^;]+)', go_str)
        for cat, desc in pattern2:
            terms.append(('', cat.strip(), desc.strip()))

    return terms


# Build gene-to-GO mapping and GO term universe
gene_go_map = defaultdict(list)
go_term_info = {}  # (category, description) -> count in background

for gene in common_genes:
    go_str = go_annotations.get(gene, '')
    # Also check TPM file GO terms
    if not go_str or go_str == '------':
        go_str = go_from_tpm_asym.get(gene, '')
    if not go_str or go_str == '------':
        go_str = go_from_tpm_sym.get(gene, '')

    terms = parse_go_terms(go_str)
    for go_id, cat, desc in terms:
        key = (cat, desc)
        gene_go_map[gene].append(key)
        if key not in go_term_info:
            go_term_info[key] = set()
        go_term_info[key].add(gene)

print(f"\nGenes with GO annotations: {len([g for g in gene_go_map if gene_go_map[g]])}")
print(f"Unique GO terms: {len(go_term_info)}")

# Enrichment analysis for common overexpressed genes at first time point (Day 10)
# Use common UP genes at Day 10 as foreground
foreground_genes = set()
for tp in time_points:
    foreground_genes |= deg_results[tp]['common_up_genes']

print(f"Total common overexpressed genes (union across time points): {len(foreground_genes)}")

# If we don't have enough common UP genes, use all UP genes
if len(foreground_genes) < 50:
    for tp in time_points:
        foreground_genes |= deg_results[tp]['all_asym_up']
        foreground_genes |= deg_results[tp]['all_sym_up']
    print(f"Extended to all upregulated genes: {len(foreground_genes)}")

N = len(common_genes)  # total genes
n = len(foreground_genes)  # foreground size

enrichment_results = []

for (cat, desc), bg_genes in go_term_info.items():
    if len(bg_genes) < 3:  # skip terms with too few genes
        continue

    K = len(bg_genes)  # genes with this term in background
    fg_with_term = foreground_genes & bg_genes
    k = len(fg_with_term)  # genes with this term in foreground

    if k == 0:
        continue

    # Fisher's exact test (one-sided, greater)
    table = [[k, n - k], [K - k, N - n - (K - k)]]
    # Make sure no negative values
    if any(v < 0 for row in table for v in row):
        continue

    odds_ratio, p_value = fisher_exact(table, alternative='greater')
    ratio = k / n if n > 0 else 0

    enrichment_results.append({
        'category': cat,
        'term': desc,
        'ratio': ratio,
        'p_value': p_value,
        'count': k,
        'bg_count': K,
    })

# Sort by p-value and select top 10 per category
enrichment_df = pd.DataFrame(enrichment_results)
if len(enrichment_df) > 0:
    enrichment_df = enrichment_df.sort_values('p_value')

    top_terms = []
    for cat in ['BP', 'CC', 'MF']:
        cat_df = enrichment_df[enrichment_df['category'] == cat].head(10)
        top_terms.append(cat_df)
        print(f"\nTop GO terms - {cat}: {len(cat_df)}")

    top_terms_df = pd.concat(top_terms, ignore_index=True)
    print(f"Total top GO terms for dot plot: {len(top_terms_df)}")
else:
    top_terms_df = pd.DataFrame()
    print("No enrichment results found")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  CREATE FIGURE
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 16))

# Layout: top row = Panel A (left ~60%) + empty/legend space
#          bottom row = Panel B (left ~40%) + Panel C (right ~60%)
gs_main = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2],
                            width_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

# ── Panel A: DEG Bar Chart ─────────────────────────────────────────────────
ax_a = fig.add_subplot(gs_main[0, :])

n_tp = len(time_points)
bar_width = 0.35
group_width = 2.5

# Colors matching the literature figure
color_common_up = '#E8A317'      # orange
color_specific_up = '#FDEAAD'    # light yellow
color_common_down = '#4A7FB5'    # blue
color_specific_down = '#B0D4E8'  # light blue

for idx, tp in enumerate(time_points):
    res = deg_results[tp]
    x_center = idx * group_width

    # Asymbiotic bar position
    x_asym = x_center - bar_width / 2 - 0.05
    # Symbiotic bar position
    x_sym = x_center + bar_width / 2 + 0.05

    # --- UP bars (positive direction) ---
    # Asymbiotic UP
    ax_a.bar(x_asym, res['common_up'], bar_width, color=color_common_up,
             edgecolor='white', linewidth=0.5)
    ax_a.bar(x_asym, res['specific_up_asym'], bar_width,
             bottom=res['common_up'], color=color_specific_up,
             edgecolor='white', linewidth=0.5)

    # Symbiotic UP
    ax_a.bar(x_sym, res['common_up'], bar_width, color=color_common_up,
             edgecolor='white', linewidth=0.5)
    ax_a.bar(x_sym, res['specific_up_sym'], bar_width,
             bottom=res['common_up'], color=color_specific_up,
             edgecolor='white', linewidth=0.5)

    # --- DOWN bars (negative direction) ---
    # Asymbiotic DOWN
    ax_a.bar(x_asym, -res['common_down'], bar_width, color=color_common_down,
             edgecolor='white', linewidth=0.5)
    ax_a.bar(x_asym, -res['specific_down_asym'], bar_width,
             bottom=-res['common_down'], color=color_specific_down,
             edgecolor='white', linewidth=0.5)

    # Symbiotic DOWN
    ax_a.bar(x_sym, -res['common_down'], bar_width, color=color_common_down,
             edgecolor='white', linewidth=0.5)
    ax_a.bar(x_sym, -res['specific_down_sym'], bar_width,
             bottom=-res['common_down'], color=color_specific_down,
             edgecolor='white', linewidth=0.5)

    # Add number labels on bars
    total_asym_up = res['common_up'] + res['specific_up_asym']
    total_sym_up = res['common_up'] + res['specific_up_sym']
    total_asym_down = res['common_down'] + res['specific_down_asym']
    total_sym_down = res['common_down'] + res['specific_down_sym']

    # Common UP label
    ax_a.text(x_asym, res['common_up'] / 2, str(res['common_up']),
              ha='center', va='center', fontsize=6, fontweight='bold')
    ax_a.text(x_sym, res['common_up'] / 2, str(res['common_up']),
              ha='center', va='center', fontsize=6, fontweight='bold')

    # Specific UP labels
    if res['specific_up_asym'] > 0:
        ax_a.text(x_asym, res['common_up'] + res['specific_up_asym'] / 2,
                  str(res['specific_up_asym']),
                  ha='center', va='center', fontsize=6)
    if res['specific_up_sym'] > 0:
        ax_a.text(x_sym, res['common_up'] + res['specific_up_sym'] / 2,
                  str(res['specific_up_sym']),
                  ha='center', va='center', fontsize=6)

    # Common DOWN label
    ax_a.text(x_asym, -res['common_down'] / 2, str(res['common_down']),
              ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    ax_a.text(x_sym, -res['common_down'] / 2, str(res['common_down']),
              ha='center', va='center', fontsize=6, fontweight='bold', color='white')

    # Specific DOWN labels
    if res['specific_down_asym'] > 0:
        ax_a.text(x_asym, -res['common_down'] - res['specific_down_asym'] / 2,
                  str(res['specific_down_asym']),
                  ha='center', va='center', fontsize=6)
    if res['specific_down_sym'] > 0:
        ax_a.text(x_sym, -res['common_down'] - res['specific_down_sym'] / 2,
                  str(res['specific_down_sym']),
                  ha='center', va='center', fontsize=6)

    # X-axis labels
    ax_a.text(x_asym, 0, 'Asymbiotic', ha='center', va='top',
              fontsize=7, rotation=45, transform=ax_a.get_xaxis_transform())
    ax_a.text(x_sym, 0, 'Symbiotic', ha='center', va='top',
              fontsize=7, rotation=45, transform=ax_a.get_xaxis_transform())

# Time point group labels
for idx, label in enumerate(time_labels):
    x_center = idx * group_width
    ax_a.text(x_center, 1.08, label, ha='center', va='bottom',
              fontsize=10, fontweight='bold',
              transform=ax_a.get_xaxis_transform())

ax_a.set_ylabel('No. of DEGs', fontsize=12, fontweight='bold')
ax_a.axhline(y=0, color='black', linewidth=0.8)
ax_a.set_xticks([])

# Add UP / DOWN labels
ylim = ax_a.get_ylim()
ax_a.text(-0.05, 0.75, 'UP', ha='right', va='center', fontsize=11,
          fontweight='bold', transform=ax_a.transAxes)
ax_a.text(-0.05, 0.25, 'DOWN', ha='right', va='center', fontsize=11,
          fontweight='bold', transform=ax_a.transAxes)

# Legend
legend_elements = [
    Patch(facecolor=color_specific_up, label='Specifically upregulated genes'),
    Patch(facecolor=color_common_up, label='Common upregulated genes'),
    Patch(facecolor=color_common_down, label='Common downregulated genes'),
    Patch(facecolor=color_specific_down, label='Specifically downregulated genes'),
]
ax_a.legend(handles=legend_elements, loc='upper right', fontsize=9,
            frameon=True, edgecolor='gray')

ax_a.text(-0.02, 1.05, 'A', transform=ax_a.transAxes, fontsize=16,
          fontweight='bold', va='bottom')

ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# ── Panel B: Heatmap with hierarchical clustering ──────────────────────────
ax_b_area = fig.add_subplot(gs_main[1, 0])
ax_b_area.set_visible(False)

# Create sub-gridspec for dendrogram + heatmap
gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1, 0],
                                         width_ratios=[0.15, 0.85], wspace=0.02)

# Subsample DEGs for heatmap if too many
max_heatmap_genes = 2000
if len(all_degs_list) > max_heatmap_genes:
    # Select genes with highest variance
    gene_var = np.var(heatmap_data, axis=1)
    top_var_idx = np.argsort(gene_var)[-max_heatmap_genes:]
    heatmap_subset = heatmap_data[top_var_idx, :]
    heatmap_genes_subset = [all_degs_list[i] for i in top_var_idx]
else:
    heatmap_subset = heatmap_data
    heatmap_genes_subset = all_degs_list

print(f"Heatmap genes: {len(heatmap_genes_subset)}")

# Perform hierarchical clustering
if len(heatmap_genes_subset) > 1:
    dist_matrix = pdist(heatmap_subset, metric='euclidean')
    linkage_matrix = linkage(dist_matrix, method='ward')

    # Dendrogram
    ax_dendro = fig.add_subplot(gs_b[0, 0])
    dendro = dendrogram(linkage_matrix, orientation='left', no_labels=True,
                        ax=ax_dendro, color_threshold=0,
                        above_threshold_color='black')
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    ax_dendro.spines['top'].set_visible(False)
    ax_dendro.spines['right'].set_visible(False)
    ax_dendro.spines['bottom'].set_visible(False)
    ax_dendro.spines['left'].set_visible(False)

    # Reorder heatmap rows by dendrogram
    order = dendro['leaves']
    heatmap_ordered = heatmap_subset[order, :]
else:
    heatmap_ordered = heatmap_subset
    ax_dendro = fig.add_subplot(gs_b[0, 0])
    ax_dendro.set_visible(False)

# Heatmap
ax_heatmap = fig.add_subplot(gs_b[0, 1])
vmax = 15
vmin = -15

# Use orange-white-blue colormap like the literature
from matplotlib.colors import LinearSegmentedColormap
colors_hm = ['#2166AC', '#67A9CF', '#D1E5F0', '#FDDBC7', '#EF8A62', '#B2182B']
cmap_custom = LinearSegmentedColormap.from_list('custom_rwb', colors_hm, N=256)

im = ax_heatmap.imshow(heatmap_ordered, aspect='auto', cmap=cmap_custom,
                        vmin=vmin, vmax=vmax, interpolation='nearest')

# X-axis labels
n_asym_cols = len(time_points)
all_col_labels = [str(tp) for tp in time_points] + [str(tp) for tp in time_points]
ax_heatmap.set_xticks(range(len(heatmap_cols)))
ax_heatmap.set_xticklabels(all_col_labels, fontsize=8)
ax_heatmap.set_yticks([])

# Add group labels below x-axis
ax_heatmap.text(n_asym_cols / 2 - 0.5, len(heatmap_ordered) + len(heatmap_ordered) * 0.08,
                'Asymbiotic', ha='center', va='top', fontsize=10, fontweight='bold')
ax_heatmap.text(n_asym_cols + n_asym_cols / 2 - 0.5, len(heatmap_ordered) + len(heatmap_ordered) * 0.08,
                'Symbiotic', ha='center', va='top', fontsize=10, fontweight='bold')

# Add "Day" label
ax_heatmap.set_xlabel('(Day)', fontsize=9)

# Colorbar
cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.5, pad=0.02)
cbar.set_label('Log$_2$FC', fontsize=10)

ax_heatmap.text(-0.3, 1.02, 'B', transform=ax_heatmap.transAxes, fontsize=16,
                fontweight='bold', va='bottom')

# ── Panel C: GO enrichment dot plot ────────────────────────────────────────
ax_c = fig.add_subplot(gs_main[1, 1])

if len(top_terms_df) > 0:
    # Prepare data for dot plot
    categories = ['BP', 'CC', 'MF']
    cat_labels = ['Biological\nProcess', 'Cellular\nComponent', 'Molecular\nFunction']

    # Get all unique terms (ordered by category and p-value)
    all_terms = top_terms_df['term'].unique()

    # Truncate long term names
    max_term_len = 40
    term_display = {}
    for t in all_terms:
        if len(t) > max_term_len:
            term_display[t] = t[:max_term_len - 3] + '...'
        else:
            term_display[t] = t

    # Create the dot plot
    for cat_idx, cat in enumerate(categories):
        cat_data = top_terms_df[top_terms_df['category'] == cat]
        for _, row in cat_data.iterrows():
            term = row['term']
            y_pos = list(all_terms).index(term)

            # Dot size proportional to ratio
            size = row['ratio'] * 800 + 20  # scale for visibility

            # Color by p-value (elimKS equivalent)
            color_val = row['p_value']

            ax_c.scatter(cat_idx, y_pos, s=size, c=[color_val],
                        cmap='Blues_r', vmin=0, vmax=0.05,
                        edgecolors='black', linewidths=0.5, zorder=3)

    ax_c.set_xticks(range(len(categories)))
    ax_c.set_xticklabels(cat_labels, fontsize=10, rotation=30, ha='right')
    ax_c.set_yticks(range(len(all_terms)))
    ax_c.set_yticklabels([term_display[t] for t in all_terms], fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_ylabel('Term', fontsize=11)
    ax_c.grid(True, alpha=0.3, linestyle='--')

    # Add size legend
    legend_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    legend_labels = ['0.2', '0.4', '0.6', '0.8', '1.0']
    size_legend_elements = []
    for s, l in zip(legend_sizes, legend_labels):
        size_legend_elements.append(
            plt.scatter([], [], s=s * 800 + 20, c='gray', edgecolors='black',
                       linewidths=0.5, label=l)
        )

    legend1 = ax_c.legend(handles=size_legend_elements, title='Ratio',
                           loc='upper right', fontsize=8, title_fontsize=9,
                           frameon=True, labelspacing=1.2,
                           bbox_to_anchor=(1.25, 1.0))
    ax_c.add_artist(legend1)

    # Add colorbar for p-value
    sm = plt.cm.ScalarMappable(cmap='Blues_r',
                                norm=plt.Normalize(vmin=0, vmax=0.05))
    sm.set_array([])
    cbar_c = plt.colorbar(sm, ax=ax_c, shrink=0.3, pad=0.15)
    cbar_c.set_label('p-value', fontsize=9)

else:
    ax_c.text(0.5, 0.5, 'No significant GO enrichment\nresults available',
              ha='center', va='center', fontsize=12, transform=ax_c.transAxes)

ax_c.text(-0.02, 1.02, 'C', transform=ax_c.transAxes, fontsize=16,
          fontweight='bold', va='bottom')

# ── Final adjustments ──────────────────────────────────────────────────────
fig.suptitle('Transcriptome analysis of asymbiotically and symbiotically germinated fungus',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('Figure_3_Transcriptome_Analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('Figure_3_Transcriptome_Analysis.pdf', dpi=300, bbox_inches='tight',
            facecolor='white')
print("\n✓ Figure saved as 'Figure_3_Transcriptome_Analysis.png' and '.pdf'")
plt.close()
