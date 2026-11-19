#!/usr/bin/env python3
"""
build_splits.py
===============
Build leakage-safe train/test splits for LatentPPI using target-protein
holdout with k-fold cross-validation.

WHY THIS EXISTS
---------------
Pairs can't be split randomly: the same protein appears in many pairs, so
a protein in both train and test lets the MLP memorize it instead of
learning the interaction. The human PPI graph is also one giant connected
component at 30% identity, so it can't be cleanly cut in half.

Solution: target-protein holdout. We hold out whole 0.3 sequence clusters
(so no homolog leaks across the split). Every pair that TOUCHES a held-out
cluster becomes a test pair for that fold; everything else is train.

KEY DESIGN DECISIONS (see conversation for rationale)
-----------------------------------------------------
1. Unit of holdout = 0.3 MMseqs2 cluster (not bare accession), so homologs
   never straddle the split.
2. Hubs (top-degree clusters) are EXCLUDED from holdout eligibility -- they
   stay in train in every fold. They are NOT removed from the data; they
   just never get to be a held-out target. Prevents one hub dragging a
   huge fraction of pairs into test and starving training.
3. Eligible clusters are stratified by (length bin x Neff bin) so each
   fold's held-out targets are representative, not accidentally all-short
   or all-shallow-MSA.
4. k-fold: every eligible cluster is a held-out target in exactly one fold.
5. "Option A" (non-disjoint) test rule: in fold k, a pair is TEST if either
   protein's cluster is a fold-k target, else TRAIN. A pair whose two
   proteins are targets in different folds is therefore tested in BOTH of
   those folds -- that's intended (each target evaluated independently).

OUTPUT
------
fold_assignments.tsv -- one row per pair, columns:
    system_name, uniprot_a, uniprot_b, fold_0 ... fold_{k-1}
each fold column is "train" or "test".

Usage:
    python build_splits.py \\
        --pairs        dataset_summary.tsv \\
        --clusters     ppi_dataset/clusters_all.tsv \\
        --neff-cache   neff_cache.json \\
        --n-folds      5 \\
        --hub-percentile 99 \\
        --out          fold_assignments.tsv
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_clusters(clusters_tsv):
    """
    Read MMseqs2 createtsv output (col0 = cluster representative = cluster id,
    col1 = member accession). Return dict: accession -> cluster_id.
    """
    df = pd.read_csv(clusters_tsv, sep="\t", header=None, names=["cluster", "protein"])
    mapping = dict(zip(df["protein"], df["cluster"]))
    log.info(f"Loaded {len(mapping)} proteins in {df['cluster'].nunique()} clusters")
    return mapping


def load_pairs(pairs_tsv):
    """Load the pair table (must have uniprot_a, uniprot_b; len_a/len_b used for binning)."""
    df = pd.read_csv(pairs_tsv, sep="\t", low_memory=False)
    df["system_name"] = df["uniprot_a"] + "_" + df["uniprot_b"]
    log.info(f"Loaded {len(df)} pairs")
    return df


def load_neff(neff_cache_path):
    """Load neff_cache.json: dict accession -> neff (may contain nulls)."""
    with open(neff_cache_path) as f:
        neff = json.load(f)
    log.info(f"Loaded Neff for {len(neff)} proteins")
    return neff


# ---------------------------------------------------------------------------
# Per-cluster properties
# ---------------------------------------------------------------------------

def compute_cluster_degree(pairs, protein_to_cluster):
    """
    Degree of a cluster = number of pairs that touch it (in THIS dataset).
    Used to identify hubs. Counted per unique cluster per pair so a
    within-cluster pair isn't double-counted.
    """
    degree = Counter()
    for a, b in zip(pairs["uniprot_a"], pairs["uniprot_b"]):
        ca, cb = protein_to_cluster.get(a), protein_to_cluster.get(b)
        clusters_in_pair = {c for c in (ca, cb) if c is not None}
        for c in clusters_in_pair:
            degree[c] += 1
    return degree


def compute_cluster_properties(pairs, protein_to_cluster, neff):
    """
    For each cluster, collect the MEDIAN length and MEDIAN Neff across all its
    member proteins that appear in the pair set. Returns dict:
        cluster_id -> {"len": median_len, "neff": median_neff}
    """
    # gather per-protein length from the pair table (each protein may appear
    # as uniprot_a in some rows and uniprot_b in others -- collect both)
    protein_len = {}
    for _, row in pairs.iterrows():
        protein_len.setdefault(row["uniprot_a"], row["len_a"])
        protein_len.setdefault(row["uniprot_b"], row["len_b"])

    # group member proteins by cluster
    cluster_members = defaultdict(set)
    for protein, cluster in protein_to_cluster.items():
        if protein in protein_len:  # only proteins actually in our pairs
            cluster_members[cluster].add(protein)

    props = {}
    for cluster, members in cluster_members.items():
        lens = [protein_len[p] for p in members if p in protein_len and pd.notna(protein_len[p])]
        neffs = [neff[p] for p in members if neff.get(p) is not None]
        props[cluster] = {
            "len": float(np.median(lens)) if lens else np.nan,
            "neff": float(np.median(neffs)) if neffs else np.nan,
        }
    return props


# ---------------------------------------------------------------------------
# Stratified assignment of eligible clusters to folds
# ---------------------------------------------------------------------------

def assign_clusters_to_folds(eligible_props, n_folds, n_bins=3, seed=0):
    """
    Assign each eligible cluster to exactly one fold (its held-out fold),
    stratified by (length bin x Neff bin) so every fold's targets are
    representative.

    Strategy: bin clusters by length tercile and Neff tercile, then within
    each (len_bin, neff_bin) cell, shuffle and round-robin the clusters
    across folds. Round-robin within each cell guarantees each fold gets a
    proportional slice of every cell.

    Returns dict: cluster_id -> fold_index
    """
    rng = np.random.default_rng(seed)

    clusters = list(eligible_props.keys())
    lens = np.array([eligible_props[c]["len"] for c in clusters])
    neffs = np.array([eligible_props[c]["neff"] for c in clusters])

    def make_bins(values, n_bins):
        """Assign each value to a quantile bin (0..n_bins-1). NaNs get their own bin."""
        bins = np.full(len(values), -1, dtype=int)  # -1 = NaN/unknown bin
        valid = ~np.isnan(values)
        if valid.sum() > 0:
            # quantile edges over valid values
            edges = np.quantile(values[valid], np.linspace(0, 1, n_bins + 1))
            # np.digitize; clip so the max value lands in the last bin
            binned = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
            bins[valid] = binned
        return bins

    len_bins = make_bins(lens, n_bins)
    neff_bins = make_bins(neffs, n_bins)

    # group clusters by (len_bin, neff_bin) cell
    cells = defaultdict(list)
    for i, c in enumerate(clusters):
        cells[(len_bins[i], neff_bins[i])].append(c)

    assignment = {}
    for cell, cell_clusters in cells.items():
        shuffled = cell_clusters.copy()
        rng.shuffle(shuffled)
        # round-robin across folds so each fold gets ~equal share of this cell
        for j, c in enumerate(shuffled):
            assignment[c] = j % n_folds

    # log fold sizes (in clusters)
    fold_counts = Counter(assignment.values())
    log.info(f"Eligible clusters per fold: {dict(sorted(fold_counts.items()))}")
    return assignment


# ---------------------------------------------------------------------------
# Derive per-pair train/test roles from cluster->fold assignment
# ---------------------------------------------------------------------------

def build_fold_columns(pairs, protein_to_cluster, cluster_fold, n_folds):
    """
    For each fold k, mark each pair train/test using the Option A rule:
        test  if either protein's cluster is a fold-k target
        train otherwise

    Hubs (and any cluster not in cluster_fold) are never a target in any
    fold, so pairs touching only hubs/non-eligible clusters are always train.

    Returns a DataFrame of fold columns aligned to `pairs`.
    """
    # invert: fold_index -> set of target clusters for that fold
    fold_targets = defaultdict(set)
    for cluster, fold in cluster_fold.items():
        fold_targets[fold].add(cluster)

    fold_cols = {f"fold_{k}": [] for k in range(n_folds)}

    for a, b in zip(pairs["uniprot_a"], pairs["uniprot_b"]):
        ca, cb = protein_to_cluster.get(a), protein_to_cluster.get(b)
        for k in range(n_folds):
            targets = fold_targets[k]
            is_test = (ca in targets) or (cb in targets)
            fold_cols[f"fold_{k}"].append("test" if is_test else "train")

    return pd.DataFrame(fold_cols, index=pairs.index)


# ---------------------------------------------------------------------------
# Post-hoc sanity checks
# ---------------------------------------------------------------------------

def report_fold_stats(pairs, fold_df, n_folds):
    """
    For each fold, log test/train pair counts, pos:neg balance, and per-tier
    negative representation -- the checks that tell you whether the split is
    actually usable (can't be enforced up front, only checked after).
    """
    has_label = "label" in pairs.columns
    has_negsrc = "neg_source" in pairs.columns

    for k in range(n_folds):
        col = f"fold_{k}"
        test_mask = fold_df[col] == "test"
        n_test = test_mask.sum()
        n_train = (~test_mask).sum()
        log.info(f"--- fold_{k}: {n_train} train / {n_test} test pairs ---")

        if has_label:
            test_pos = ((pairs["label"] == 1) & test_mask).sum()
            test_neg = ((pairs["label"] == 0) & test_mask).sum()
            log.info(f"    test pos/neg: {test_pos}/{test_neg}")

        if has_negsrc:
            tier_counts = pairs.loc[test_mask & (pairs.get("label", 1) == 0), "neg_source"].value_counts()
            log.info(f"    test neg tiers: {dict(tier_counts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    protein_to_cluster = load_clusters(args.clusters)
    pairs = load_pairs(args.pairs)
    neff = load_neff(args.neff_cache)

    # --- degree -> identify hubs ---
    degree = compute_cluster_degree(pairs, protein_to_cluster)
    degrees = np.array(list(degree.values()))
    hub_cutoff = np.percentile(degrees, args.hub_percentile)
    hub_clusters = {c for c, d in degree.items() if d >= hub_cutoff}
    log.info(
        f"Hub cutoff at {args.hub_percentile}th percentile = degree {hub_cutoff:.0f}; "
        f"{len(hub_clusters)} clusters excluded from holdout eligibility (stay in train)"
    )

    # --- per-cluster length/Neff for stratification ---
    cluster_props = compute_cluster_properties(pairs, protein_to_cluster, neff)

    # eligible clusters = have properties AND are not hubs
    eligible_props = {
        c: p for c, p in cluster_props.items()
        if c not in hub_clusters
    }
    log.info(f"{len(eligible_props)} clusters eligible as held-out targets")

    # --- assign eligible clusters to folds (stratified) ---
    cluster_fold = assign_clusters_to_folds(
        eligible_props, n_folds=args.n_folds, n_bins=args.n_bins, seed=args.seed
    )

    # --- derive per-pair fold roles ---
    fold_df = build_fold_columns(pairs, protein_to_cluster, cluster_fold, args.n_folds)

    # --- sanity checks ---
    report_fold_stats(pairs, fold_df, args.n_folds)
    tprot = set(pairs.uniprot_a) | set(pairs.uniprot_b)
    unmapped = prot - set(protein_to_cluster)
    print(f"{len(unmapped)} of {len(prot)} proteins have no cluster assignment")

    # --- write output ---
    out = pd.concat(
        [pairs[["system_name", "uniprot_a", "uniprot_b"]].reset_index(drop=True),
         fold_df.reset_index(drop=True)],
        axis=1,
    )
    out.to_csv(args.out, sep="\t", index=False)
    log.info(f"Wrote {len(out)} pairs x {args.n_folds} folds to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build target-holdout k-fold splits for LatentPPI")
    ap.add_argument("--pairs", required=True, help="dataset_summary.tsv (uniprot_a/b, len_a/b, label, neg_source)")
    ap.add_argument("--clusters", required=True, help="MMseqs2 clusters_all.tsv (0.3 global clustering)")
    ap.add_argument("--neff-cache", required=True, help="neff_cache.json (accession -> Neff)")
    ap.add_argument("--n-folds", type=int, default=5, help="number of CV folds (default: 5)")
    ap.add_argument("--hub-percentile", type=float, default=99,
                    help="clusters at/above this degree percentile are excluded from holdout (default: 99)")
    ap.add_argument("--n-bins", type=int, default=3,
                    help="number of quantile bins per stratification axis, length & Neff (default: 3)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible fold assignment")
    ap.add_argument("--out", default="fold_assignments.tsv", help="output TSV path")
    args = ap.parse_args()
    main(args)