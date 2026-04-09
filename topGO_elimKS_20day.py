#!/usr/bin/env python3
"""
GO enrichment analysis using elimKS method for 20-day symbiotic-specific genes.

Steps:
1. Identify genes specifically UP-regulated at 20 days in symbiotic (MS20) but NOT in non-symbiotic (M20)
2. Identify genes specifically DOWN-regulated at 20 days in non-symbiotic (M20) but NOT in symbiotic (MS20)
3. Combine these two gene sets (union)
4. Extract background genes and GO annotations from the gene count matrix
5. Perform GO enrichment using the elimKS algorithm (elimination + Kolmogorov-Smirnov test)
"""

import csv
import re
import sys
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from goatools.obo_parser import GODag

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OBO_FILE = os.environ.get("GO_OBO_FILE", "/tmp/go-ontology/src/ontology/go-edit.obo")
ANNOT_FILE = os.path.join(DATA_DIR, "gene.count.matrix.annot.xls")

# DEG files
MS20_FILE = os.path.join(DATA_DIR, "MS20VSM5.csv")  # Symbiotic 20d vs 5d
M20_FILE = os.path.join(DATA_DIR, "M20VSM5.csv")    # Non-symbiotic 20d vs 5d

# Output files
OUTPUT_DIR = DATA_DIR
GENE_LIST_FILE = os.path.join(OUTPUT_DIR, "20day_specific_genes.csv")
GO_RESULT_BP = os.path.join(OUTPUT_DIR, "GO_enrichment_elimKS_BP.csv")
GO_RESULT_MF = os.path.join(OUTPUT_DIR, "GO_enrichment_elimKS_MF.csv")
GO_RESULT_CC = os.path.join(OUTPUT_DIR, "GO_enrichment_elimKS_CC.csv")
GO_RESULT_ALL = os.path.join(OUTPUT_DIR, "GO_enrichment_elimKS_all.csv")

# ============================================================
# Step 1 & 2: Identify target genes
# ============================================================
def read_deg_file(filepath):
    """Read a DEG CSV file and return sets of up/down regulated genes."""
    up_genes = set()
    down_genes = set()
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene_id = row["Gene ID"]
            sig = row.get("Significant", "")
            regulate = row.get("Regulate", "")
            if sig.lower() == "yes":
                if regulate.lower() == "up":
                    up_genes.add(gene_id)
                elif regulate.lower() == "down":
                    down_genes.add(gene_id)
    return up_genes, down_genes


def get_deg_pvalues(filepath):
    """Read a DEG CSV file and return gene p-values."""
    pvalues = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene_id = row["Gene ID"]
            try:
                pval = float(row["Pvalue"])
            except (ValueError, KeyError):
                pval = 1.0
            pvalues[gene_id] = pval
    return pvalues


print("=" * 70)
print("Step 1: Reading DEG files...")
print("=" * 70)

ms20_up, ms20_down = read_deg_file(MS20_FILE)
m20_up, m20_down = read_deg_file(M20_FILE)

print(f"  MS20 (Symbiotic 20d): {len(ms20_up)} UP, {len(ms20_down)} DOWN genes")
print(f"  M20 (Non-symbiotic 20d): {len(m20_up)} UP, {len(m20_down)} DOWN genes")

# Symbiotic-specific UP genes: UP in MS20 but NOT UP in M20
sym_specific_up = ms20_up - m20_up
print(f"\n  Symbiotic-specific UP genes (MS20 UP - M20 UP): {len(sym_specific_up)}")

# Non-symbiotic-specific DOWN genes: DOWN in M20 but NOT DOWN in MS20
nonsym_specific_down = m20_down - ms20_down
print(f"  Non-symbiotic-specific DOWN genes (M20 DOWN - MS20 DOWN): {len(nonsym_specific_down)}")

# Union of both sets
target_genes = sym_specific_up | nonsym_specific_down
print(f"\n  Combined target gene set (union): {len(target_genes)} genes")

# ============================================================
# Step 3: Get background genes and GO annotations
# ============================================================
print("\n" + "=" * 70)
print("Step 2: Reading background genes and GO annotations...")
print("=" * 70)

gene2go = defaultdict(set)  # gene -> set of GO terms
go2gene = defaultdict(set)  # GO term -> set of genes
all_genes = set()
go_term_info = {}  # GO term -> (namespace, name)

with open(ANNOT_FILE, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for row in reader:
        if len(row) < 34:
            continue
        gene_id = row[0]
        all_genes.add(gene_id)
        go_col = row[33]  # GO annotation column
        if go_col and go_col.strip():
            # Parse GO annotations: "GO:0016787(molecular_function:hydrolase activity); ..."
            entries = go_col.split(";")
            for entry in entries:
                entry = entry.strip()
                m = re.match(r'(GO:\d+)\((\w+):(.+)\)', entry)
                if m:
                    go_id = m.group(1)
                    ns = m.group(2)
                    name = m.group(3)
                    gene2go[gene_id].add(go_id)
                    go2gene[go_id].add(gene_id)
                    # Map namespace to short form
                    ns_short = {"biological_process": "BP",
                                "molecular_function": "MF",
                                "cellular_component": "CC"}.get(ns, ns)
                    go_term_info[go_id] = (ns_short, name)

print(f"  Total background genes: {len(all_genes)}")
print(f"  Genes with GO annotations: {len(gene2go)}")
print(f"  Unique GO terms: {len(go2gene)}")
print(f"  Target genes with GO: {len(target_genes & set(gene2go.keys()))}")

# Save target gene list
with open(GENE_LIST_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Gene_ID", "Category"])
    for g in sorted(target_genes):
        if g in sym_specific_up:
            cat = "Symbiotic_specific_UP"
        else:
            cat = "NonSymbiotic_specific_DOWN"
        writer.writerow([g, cat])
print(f"  Target gene list saved to: {GENE_LIST_FILE}")

# Get p-values for all genes from both DEG analyses
ms20_pvals = get_deg_pvalues(MS20_FILE)
m20_pvals = get_deg_pvalues(M20_FILE)

# Create combined gene p-values for target genes
# For symbiotic-specific UP genes: use MS20 p-value
# For non-symbiotic-specific DOWN genes: use M20 p-value
gene_pvalues = {}
for gene in target_genes:
    if gene in sym_specific_up and gene in ms20_pvals:
        gene_pvalues[gene] = ms20_pvals[gene]
    elif gene in nonsym_specific_down and gene in m20_pvals:
        gene_pvalues[gene] = m20_pvals[gene]

# ============================================================
# Step 4: Load GO DAG
# ============================================================
print("\n" + "=" * 70)
print("Step 3: Loading GO DAG...")
print("=" * 70)

dag = GODag(OBO_FILE)
print(f"  GO DAG loaded: {len(dag)} terms")

# ============================================================
# Step 5: Implement elimKS algorithm
# ============================================================
print("\n" + "=" * 70)
print("Step 4: Running elimKS GO enrichment analysis...")
print("=" * 70)


def get_all_ancestors(go_id, dag):
    """Get all ancestor GO terms of a given GO term."""
    ancestors = set()
    if go_id not in dag:
        return ancestors
    term = dag[go_id]
    parents = term.parents
    for parent in parents:
        ancestors.add(parent.id)
        ancestors.update(get_all_ancestors(parent.id, dag))
    return ancestors


def get_all_descendants(go_id, dag):
    """Get all descendant GO terms of a given GO term."""
    descendants = set()
    if go_id not in dag:
        return descendants
    term = dag[go_id]
    children = term.children
    for child in children:
        descendants.add(child.id)
        descendants.update(get_all_descendants(child.id, dag))
    return descendants


def propagate_annotations(gene2go_input, dag):
    """
    Propagate GO annotations up the DAG.
    Each gene annotated to a term is also annotated to all ancestor terms.
    """
    gene2go_prop = defaultdict(set)
    for gene, go_terms in gene2go_input.items():
        for go_id in go_terms:
            gene2go_prop[gene].add(go_id)
            ancestors = get_all_ancestors(go_id, dag)
            gene2go_prop[gene].update(ancestors)
    return gene2go_prop


def build_go2gene_from_gene2go(gene2go_map):
    """Build reverse mapping from GO term to genes."""
    go2gene_map = defaultdict(set)
    for gene, go_terms in gene2go_map.items():
        for go_id in go_terms:
            go2gene_map[go_id].add(gene)
    return go2gene_map


def get_go_level(go_id, dag, cache=None):
    """Get the depth/level of a GO term in the DAG."""
    if cache is None:
        cache = {}
    if go_id in cache:
        return cache[go_id]
    if go_id not in dag:
        cache[go_id] = 0
        return 0
    term = dag[go_id]
    if not term.parents:
        cache[go_id] = 0
        return 0
    level = 1 + max(get_go_level(p.id, dag, cache) for p in term.parents)
    cache[go_id] = level
    return level


def run_elimKS(target_genes, all_genes, gene2go_orig, dag, gene_pvalues, ontology="BP",
               pvalue_threshold=0.01, min_annotated=5):
    """
    Run the elimKS algorithm.

    The elimination algorithm processes GO terms from the most specific (deepest)
    to the most general (root). For each term, it performs a KS test comparing
    the score distribution of annotated vs non-annotated genes.
    When a significant term is found, genes annotated to it are removed from
    parent terms to reduce redundancy.

    Parameters:
    -----------
    target_genes : set - genes of interest
    all_genes : set - background genes
    gene2go_orig : dict - gene to GO term mappings (original, not propagated)
    dag : GODag - GO DAG
    gene_pvalues : dict - gene ID to p-value mapping for target genes
        (from differential expression analysis; lower = more significant)
    ontology : str - "BP", "MF", or "CC"
    pvalue_threshold : float - significance threshold for elimination
    min_annotated : int - minimum number of annotated genes for a GO term

    Returns:
    --------
    list of (go_id, go_name, annotated, significant, expected, elimKS_pvalue, ontology)
    """
    ns_map = {"BP": "biological_process", "MF": "molecular_function",
              "CC": "cellular_component"}
    target_ns = ns_map[ontology]

    # Filter gene2go for genes in our universe
    gene2go_filtered = {g: gos for g, gos in gene2go_orig.items() if g in all_genes}

    # Propagate annotations
    print(f"  [{ontology}] Propagating GO annotations...")
    gene2go_prop = propagate_annotations(gene2go_filtered, dag)
    go2gene_prop = build_go2gene_from_gene2go(gene2go_prop)

    # Filter GO terms by ontology and minimum annotation count
    valid_go_terms = set()
    for go_id, genes in go2gene_prop.items():
        if go_id in dag and dag[go_id].namespace == target_ns:
            if len(genes) >= min_annotated:
                valid_go_terms.add(go_id)

    print(f"  [{ontology}] Valid GO terms with >= {min_annotated} annotations: {len(valid_go_terms)}")

    if not valid_go_terms:
        return []

    # Assign gene scores: use actual p-values from DEG analysis for target genes
    # Non-target genes get score 1.0 (least significant)
    # This provides a continuous distribution for the KS test
    gene_scores = {}
    for gene in all_genes:
        if gene in target_genes:
            # Use actual p-value from the relevant DEG analysis
            pval = 1.0
            if gene in gene_pvalues:
                pval = gene_pvalues[gene]
            gene_scores[gene] = pval
        else:
            gene_scores[gene] = 1.0  # non-target gene = not significant

    # Sort GO terms by depth (deepest first for elimination)
    level_cache = {}
    go_levels = [(go_id, get_go_level(go_id, dag, level_cache)) for go_id in valid_go_terms]
    go_levels.sort(key=lambda x: -x[1])  # deepest first

    # Elimination algorithm
    # Track which genes to remove from parent terms
    genes_to_remove = defaultdict(set)  # go_id -> set of genes to remove
    results = []

    # Current gene annotations (mutable - genes will be removed during elimination)
    current_go2gene = {go_id: set(genes) for go_id, genes in go2gene_prop.items()}

    total = len(go_levels)
    for idx, (go_id, level) in enumerate(go_levels):
        if (idx + 1) % 500 == 0:
            print(f"  [{ontology}] Processing {idx + 1}/{total} GO terms...")

        # Get current annotated genes (after elimination from children)
        annotated_genes = current_go2gene.get(go_id, set()) & all_genes

        if len(annotated_genes) < min_annotated:
            continue

        # Get scores for annotated and non-annotated genes
        ann_scores = [gene_scores[g] for g in annotated_genes if g in gene_scores]
        non_ann_genes = all_genes - annotated_genes
        non_ann_scores = [gene_scores[g] for g in non_ann_genes if g in gene_scores]

        if len(ann_scores) < 2 or len(non_ann_scores) < 2:
            continue

        # KS test: test if annotated genes have lower scores (smaller p-values = more target genes)
        # alternative='greater' means annotated genes CDF is above non-annotated CDF
        # i.e., annotated genes are shifted toward lower values (more significant)
        ks_stat, ks_pvalue = stats.ks_2samp(ann_scores, non_ann_scores, alternative='greater')

        # Count statistics
        n_annotated = len(annotated_genes)
        n_significant = len(annotated_genes & target_genes)
        expected = n_annotated * len(target_genes) / len(all_genes) if len(all_genes) > 0 else 0

        go_name = dag[go_id].name if go_id in dag else go_id
        results.append((go_id, go_name, n_annotated, n_significant, expected,
                         ks_pvalue, ontology))

        # Elimination: if significant, remove annotated genes from all parent terms
        if ks_pvalue < pvalue_threshold:
            sig_genes_in_term = annotated_genes & target_genes
            if sig_genes_in_term:
                ancestors = get_all_ancestors(go_id, dag)
                for anc_id in ancestors:
                    if anc_id in current_go2gene:
                        current_go2gene[anc_id] -= sig_genes_in_term

    # Sort by p-value
    results.sort(key=lambda x: x[5])
    print(f"  [{ontology}] Completed. Found {sum(1 for r in results if r[5] < 0.05)} significant terms (p < 0.05)")
    return results


# Run for all three ontologies
all_results = []
for ont in ["BP", "MF", "CC"]:
    results = run_elimKS(target_genes, all_genes, dict(gene2go), dag,
                         gene_pvalues, ontology=ont)
    all_results.extend(results)

# ============================================================
# Step 6: Save results
# ============================================================
print("\n" + "=" * 70)
print("Step 5: Saving results...")
print("=" * 70)

# Save separate files for each ontology
for ont, outfile in [("BP", GO_RESULT_BP), ("MF", GO_RESULT_MF), ("CC", GO_RESULT_CC)]:
    ont_results = [r for r in all_results if r[6] == ont]
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GO_ID", "GO_Term", "Annotated", "Significant",
                          "Expected", "elimKS_pvalue", "Ontology"])
        for r in ont_results:
            writer.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.2f}", f"{r[5]:.6e}", r[6]])
    print(f"  {ont} results saved to: {outfile} ({len(ont_results)} terms)")

# Save combined file
with open(GO_RESULT_ALL, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["GO_ID", "GO_Term", "Annotated", "Significant",
                      "Expected", "elimKS_pvalue", "Ontology"])
    for r in sorted(all_results, key=lambda x: x[5]):
        writer.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.2f}", f"{r[5]:.6e}", r[6]])
print(f"  All results saved to: {GO_RESULT_ALL} ({len(all_results)} terms)")

# ============================================================
# Step 7: Print top results summary
# ============================================================
print("\n" + "=" * 70)
print("Summary: Top 20 most significant GO terms (elimKS)")
print("=" * 70)
print(f"{'GO_ID':<12} {'Ontology':<5} {'Annotated':>9} {'Significant':>11} {'P-value':>12} {'GO_Term'}")
print("-" * 90)
for r in sorted(all_results, key=lambda x: x[5])[:20]:
    print(f"{r[0]:<12} {r[6]:<5} {r[2]:>9} {r[3]:>11} {r[5]:>12.6e} {r[1][:50]}")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)
print(f"\nTarget genes: {len(target_genes)}")
print(f"  - Symbiotic-specific UP (MS20): {len(sym_specific_up)}")
print(f"  - Non-symbiotic-specific DOWN (M20): {len(nonsym_specific_down)}")
print(f"Background genes: {len(all_genes)}")
print(f"\nOutput files:")
print(f"  - Target gene list: {GENE_LIST_FILE}")
print(f"  - GO enrichment (BP): {GO_RESULT_BP}")
print(f"  - GO enrichment (MF): {GO_RESULT_MF}")
print(f"  - GO enrichment (CC): {GO_RESULT_CC}")
print(f"  - GO enrichment (All): {GO_RESULT_ALL}")
