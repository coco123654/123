#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
萌发菌（Mycena）基因表达可视化脚本
Generate expression visualization comparing symbiotic vs non-symbiotic
Mycena (germination fungi) across 5–50 days.

Usage:
    python generate_expression_figure.py

Input files (place in the same directory):
    - 萌发菌与天麻共生5-50天的表达量总表.csv
    - 萌发菌自己非共生5-50天的表达量总表.csv
    - 萌发菌与天麻共生5-50天的基因计数表达定量注释结果表.csv
    - 萌发菌自己非共生5-50天的基因计数表达定量注释结果表.csv

Output:
    - expression_figure.png  (high-resolution publication-quality figure)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
TIME_POINTS = [5, 10, 20, 30, 40, 50]

# File names – adjust if your actual file names differ slightly
FILE_SYM_EXPR  = "萌发菌与天麻共生5-50天的表达量总表.csv"
FILE_NON_EXPR  = "萌发菌自己非共生5-50天的表达量总表.csv"
FILE_SYM_ANNOT = "萌发菌与天麻共生5-50天的基因计数表达定量注释结果表.csv"
FILE_NON_ANNOT = "萌发菌自己非共生5-50天的基因计数表达定量注释结果表.csv"

# Gene categories to display (order matters)
CATEGORY_ORDER = [
    "MFS transporter",
    "Nitrogen metabolism",
    "IAA synthesis",
    "GPCR",
    "Lignin-modifying",
    "Signal transduction",
]

# Color palette
COLOR_SYM     = "#D62728"   # red  – symbiotic
COLOR_NON     = "#1F77B4"   # blue – non-symbiotic
COLOR_FILL_S  = "#FFAAAA"
COLOR_FILL_N  = "#AAC8E8"

# ─── Utility helpers ─────────────────────────────────────────────────────────

def find_file(name):
    """Locate a file by name (supports xlsx as fallback)."""
    for ext in ("", ".csv", ".xlsx", ".xls"):
        path = name if ext == "" else name.replace(".csv", ext)
        if os.path.exists(path):
            return path
    return None


def read_table(path):
    """Read CSV or Excel into DataFrame, stripping BOM if present."""
    if path is None:
        return None
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def fpkm_cols(df):
    """Return ordered list of FPKM column names found in df."""
    cols = []
    for tp in TIME_POINTS:
        for prefix in ("FPKM_", "fpkm_", "RPKM_", "TPM_", "tpm_"):
            cname = f"{prefix}{tp}d"
            if cname in df.columns:
                cols.append(cname)
                break
        else:
            # fallback: look for any column containing the time-point number
            candidates = [c for c in df.columns
                          if str(tp) in c and c not in cols]
            if candidates:
                cols.append(candidates[0])
    return cols


def count_cols(df):
    """Return ordered list of Count column names found in df."""
    cols = []
    for tp in TIME_POINTS:
        for prefix in ("Count_", "count_", "Counts_"):
            cname = f"{prefix}{tp}d"
            if cname in df.columns:
                cols.append(cname)
                break
    return cols


def get_gene_id_col(df):
    for c in ("Gene_ID", "GeneID", "gene_id", "ID"):
        if c in df.columns:
            return c
    return df.columns[0]


def get_category_col(df):
    for c in ("Category", "category", "Type", "Function", "Class"):
        if c in df.columns:
            return c
    return None


# ─── Load data ────────────────────────────────────────────────────────────────

def load_data():
    sym_expr  = read_table(find_file(FILE_SYM_EXPR))
    non_expr  = read_table(find_file(FILE_NON_EXPR))
    sym_annot = read_table(find_file(FILE_SYM_ANNOT))
    non_annot = read_table(find_file(FILE_NON_ANNOT))

    missing = []
    for name, df in [("symbiotic expression",     sym_expr),
                     ("non-symbiotic expression",  non_expr),
                     ("symbiotic annotation",      sym_annot),
                     ("non-symbiotic annotation",  non_annot)]:
        if df is None:
            missing.append(name)

    if missing:
        print(f"[Warning] Could not find: {', '.join(missing)}")
        print("  Using available tables only.")

    return sym_expr, non_expr, sym_annot, non_annot


# ─── Merge expression + annotation ───────────────────────────────────────────

def merge_tables(expr_df, annot_df):
    """Merge expression table with annotation on Gene_ID."""
    if expr_df is None and annot_df is None:
        return None

    if expr_df is None:
        return annot_df
    if annot_df is None:
        return expr_df

    id_col_e = get_gene_id_col(expr_df)
    id_col_a = get_gene_id_col(annot_df)

    # Keep annotation columns that are not already in expr_df
    annot_keep = [id_col_a] + [c for c in annot_df.columns
                                if c not in expr_df.columns and c != id_col_a]
    merged = expr_df.merge(annot_df[annot_keep],
                           left_on=id_col_e, right_on=id_col_a, how="left")
    return merged


# ─── Build per-gene summary across time points ───────────────────────────────

def build_gene_profiles(sym_df, non_df):
    """
    Returns a dict: gene_id -> {
        'gene_name': str,
        'category': str,
        'sym_fpkm':  list[float],   # mean over replicates per time point
        'non_fpkm':  list[float],
    }
    """
    profiles = {}

    for condition, df in [("sym", sym_df), ("non", non_df)]:
        if df is None:
            continue
        id_col  = get_gene_id_col(df)
        cat_col = get_category_col(df)
        name_col = "Gene_Name" if "Gene_Name" in df.columns else None
        f_cols  = fpkm_cols(df)

        if not f_cols:
            # fall back to count columns normalised to pseudo-FPKM
            f_cols = count_cols(df)

        for _, row in df.iterrows():
            gene_id = str(row[id_col])
            if gene_id not in profiles:
                profiles[gene_id] = {
                    "gene_name": row[name_col] if name_col else gene_id,
                    "category":  row[cat_col]  if cat_col  else "Other",
                    "sym_fpkm":  [np.nan] * len(TIME_POINTS),
                    "non_fpkm":  [np.nan] * len(TIME_POINTS),
                }
            vals = [float(row[c]) if not pd.isna(row[c]) else np.nan
                    for c in f_cols]
            key = "sym_fpkm" if condition == "sym" else "non_fpkm"
            profiles[gene_id][key] = vals

    return profiles


# ─── Figure: multi-panel line-plot per category ──────────────────────────────

def plot_expression_by_category(profiles, output="expression_figure.png"):
    """
    For each gene category, plot a panel showing mean FPKM ± SD across
    selected genes over time, comparing symbiotic vs non-symbiotic.
    """
    # Aggregate by category
    cat_data = {cat: {"sym": [], "non": []} for cat in CATEGORY_ORDER}
    cat_data["Other"] = {"sym": [], "non": []}

    for gid, info in profiles.items():
        cat = info["category"] if info["category"] in cat_data else "Other"
        if not np.all(np.isnan(info["sym_fpkm"])):
            cat_data[cat]["sym"].append(info["sym_fpkm"])
        if not np.all(np.isnan(info["non_fpkm"])):
            cat_data[cat]["non"].append(info["non_fpkm"])

    # Determine which categories have data
    cats_with_data = [c for c in CATEGORY_ORDER
                      if cat_data[c]["sym"] or cat_data[c]["non"]]
    n = len(cats_with_data)
    if n == 0:
        print("No data found for plotting.")
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    fig.patch.set_facecolor("white")

    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    x = np.array(TIME_POINTS)

    for idx, cat in enumerate(cats_with_data):
        row_i = idx // ncols
        col_i = idx % ncols
        ax = axes[row_i, col_i]

        sym_mat = np.array(cat_data[cat]["sym"], dtype=float)
        non_mat = np.array(cat_data[cat]["non"], dtype=float)

        def plot_line(mat, color, fill_color, label):
            if mat.size == 0:
                return
            mean_ = np.nanmean(mat, axis=0)
            std_  = np.nanstd(mat,  axis=0)
            ax.plot(x, mean_, color=color, linewidth=2.2,
                    marker="o", markersize=6, label=label, zorder=3)
            ax.fill_between(x, mean_ - std_, mean_ + std_,
                            color=fill_color, alpha=0.30, zorder=2)
            # Add significance markers between sym and non at each tp
            if sym_mat.size > 0 and non_mat.size > 0:
                return mean_, std_
            return mean_, std_

        sym_mean = non_mean = None
        if sym_mat.size > 0:
            sym_mean, sym_std = plot_line(sym_mat, COLOR_SYM, COLOR_FILL_S,
                                          "Symbiotic (与天麻共生)")
        if non_mat.size > 0:
            non_mean, non_std = plot_line(non_mat, COLOR_NON, COLOR_FILL_N,
                                          "Non-symbiotic (自身非共生)")

        # Significance markers
        if sym_mean is not None and non_mean is not None:
            y_top = max(np.nanmax(sym_mat), np.nanmax(non_mat))
            for tp_idx, tp in enumerate(TIME_POINTS):
                s_vals = sym_mat[:, tp_idx]
                n_vals = non_mat[:, tp_idx]
                s_vals = s_vals[~np.isnan(s_vals)]
                n_vals = n_vals[~np.isnan(n_vals)]
                if len(s_vals) < 2 or len(n_vals) < 2:
                    fc = (np.nanmean(s_vals) / np.nanmean(n_vals)
                          if np.nanmean(n_vals) > 0 else 1)
                    marker = "**" if fc > 5 else ("*" if fc > 2 else "")
                else:
                    _, pval = stats.ttest_ind(s_vals, n_vals)
                    marker = ("***" if pval < 0.001
                              else "**"  if pval < 0.01
                              else "*"   if pval < 0.05
                              else "")
                if marker:
                    ax.text(tp, y_top * 1.05, marker,
                            ha="center", va="bottom", fontsize=9,
                            color="#444444")

        ax.set_title(cat, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Days post-inoculation (天)", fontsize=10)
        ax.set_ylabel("FPKM", fontsize=10)
        ax.set_xticks(TIME_POINTS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{int(v)}" if v >= 10 else f"{v:.1f}"))

        if idx == 0:
            ax.legend(fontsize=9, framealpha=0.7,
                      loc="upper left", borderpad=0.5)

    # Hide unused axes
    for idx in range(len(cats_with_data), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # Global title
    fig.suptitle(
        "萌发菌（Mycena）基因表达量动态\n"
        "Gene Expression Dynamics in Symbiotic vs Non-symbiotic Mycena (5–50 days)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"[OK] Figure saved to: {output}")
    plt.close()


# ─── Figure: individual gene line-plots (key genes) ─────────────────────────

def plot_key_genes(profiles, key_genes=None, output="key_genes_expression.png"):
    """
    Plot individual line graphs for selected key genes (e.g. MyNrt, MyNir,
    MyAmid, GPCR_01 …).
    """
    if key_genes is None:
        key_genes = ["MyNrt", "MyNir", "MyAmid", "GPCR_01", "LAC_01", "NIT_01"]

    # Match by gene_name
    name_to_gid = {info["gene_name"]: gid
                   for gid, info in profiles.items()}

    selected = []
    for name in key_genes:
        gid = name_to_gid.get(name)
        if gid:
            selected.append((name, profiles[gid]))

    if not selected:
        print("[Info] No matching key genes found, skipping key-gene plot.")
        return

    n = len(selected)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor("white")

    x = np.array(TIME_POINTS)

    for idx, (name, info) in enumerate(selected):
        ax = axes[idx // ncols][idx % ncols]

        sym = np.array(info["sym_fpkm"], dtype=float)
        non = np.array(info["non_fpkm"], dtype=float)

        ax.plot(x, sym, color=COLOR_SYM, linewidth=2.5,
                marker="o", markersize=7,
                label="Symbiotic (共生)", zorder=3)
        ax.plot(x, non, color=COLOR_NON, linewidth=2.5,
                marker="s", markersize=7, linestyle="--",
                label="Non-symbiotic (非共生)", zorder=3)

        # Shade between lines
        ax.fill_between(x, non, sym,
                        where=(sym >= non),
                        interpolate=True,
                        color=COLOR_FILL_S, alpha=0.25)

        # Fold-change annotation at day 50
        fc = sym[-1] / non[-1] if non[-1] > 0 else float("inf")
        ax.annotate(f"FC={fc:.1f}×",
                    xy=(x[-1], sym[-1]), xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=8, color=COLOR_SYM,
                    va="center")

        ax.set_title(f"$\\it{{{name}}}$", fontsize=12, fontweight="bold")
        ax.set_xlabel("Days (天)", fontsize=9)
        ax.set_ylabel("FPKM", fontsize=9)
        ax.set_xticks(TIME_POINTS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

        if idx == 0:
            ax.legend(fontsize=8, framealpha=0.7, loc="upper left")

    for idx in range(len(selected), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "萌发菌关键基因表达量（共生 vs 非共生）\n"
        "Key Gene Expression in Mycena: Symbiotic vs Non-symbiotic",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"[OK] Figure saved to: {output}")
    plt.close()


# ─── Figure: category-level bar chart (mean FPKM at each time point) ─────────

def plot_category_bar(profiles, output="category_expression_bar.png"):
    """
    Grouped bar chart: for each time point, compare mean FPKM between
    symbiotic and non-symbiotic per category.
    """
    records = []
    for gid, info in profiles.items():
        cat = info["category"]
        for i, tp in enumerate(TIME_POINTS):
            records.append({
                "Category": cat,
                "Day": tp,
                "Symbiotic":     info["sym_fpkm"][i],
                "Non-symbiotic": info["non_fpkm"][i],
            })

    df = pd.DataFrame(records)
    df = df[df["Category"].isin(CATEGORY_ORDER)]

    cats = [c for c in CATEGORY_ORDER if c in df["Category"].values]
    n = len(cats)
    if n == 0:
        return

    fig, axes = plt.subplots(1, len(TIME_POINTS),
                             figsize=(3 * len(TIME_POINTS), 5),
                             sharey=False)
    fig.patch.set_facecolor("white")

    for tp_idx, tp in enumerate(TIME_POINTS):
        ax = axes[tp_idx]
        sub = df[df["Day"] == tp].groupby("Category").mean(numeric_only=True)
        sub = sub.reindex(cats)

        x_pos = np.arange(len(cats))
        w = 0.35
        ax.bar(x_pos - w/2, sub["Symbiotic"],     width=w,
               color=COLOR_SYM, alpha=0.80, label="Sym")
        ax.bar(x_pos + w/2, sub["Non-symbiotic"],  width=w,
               color=COLOR_NON, alpha=0.80, label="Non-sym")

        ax.set_title(f"Day {tp}", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace(" ", "\n") for c in cats],
                           fontsize=7, rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if tp_idx == 0:
            ax.set_ylabel("Mean FPKM", fontsize=9)
            ax.legend(fontsize=8, framealpha=0.7)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "各时间点基因分类平均表达量（共生 vs 非共生）\n"
        "Mean FPKM per Gene Category at Each Time Point",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"[OK] Figure saved to: {output}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("萌发菌基因表达可视化 / Mycena Expression Visualization")
    print("=" * 60)

    # --- Load and merge ---
    sym_expr, non_expr, sym_annot, non_annot = load_data()

    sym_df  = merge_tables(sym_expr,  sym_annot)
    non_df  = merge_tables(non_expr,  non_annot)

    if sym_df is None and non_df is None:
        print("[Error] No data files could be loaded. "
              "Please place the four CSV/Excel files in the current directory.")
        sys.exit(1)

    # --- Build profiles ---
    profiles = build_gene_profiles(sym_df, non_df)
    print(f"[Info] Loaded {len(profiles)} genes.")

    # --- Generate figures ---
    plot_expression_by_category(profiles,
                                output="expression_figure.png")

    plot_key_genes(profiles,
                   key_genes=["MyNrt", "MyNir", "MyAmid",
                               "GPCR_01", "LAC_01", "NIT_01",
                               "MAP_01",  "POD_01"],
                   output="key_genes_expression.png")

    plot_category_bar(profiles,
                      output="category_expression_bar.png")

    print("\nDone! Generated figures:")
    print("  • expression_figure.png        – category-level line plots")
    print("  • key_genes_expression.png     – individual key-gene line plots")
    print("  • category_expression_bar.png  – grouped bar charts")


if __name__ == "__main__":
    main()
