"""
PPI Dataset Curator — HINT + Negatome + Compartment + Complex-Decoy
=====================================================================
Builds a balanced PPI dataset for latent MLP training, with negatives drawn
from FOUR independent tiers instead of one contaminated cross-sampling
scheme. See negative_builders.py for the rationale behind each tier.

  Positives — HINT human binary HQ interactions
               sequences fetched from UniProt
               filtered by combined length < 700 aa
               filtered by 30% sequence identity (within-set + vs negatives)

  Negatives — Tier 1: negatome_direct        (validated, highest confidence)
              Tier 2: complex_decoy          (Predictomes-style, hard negatives)
              Tier 3: compartment_exclusion  (bulk volume, easy negatives)
              Tier 4: degree_matched_cross   (bulk fill, degree-bias-corrected)

Requirements:
    pip install requests pandas numpy tqdm biopython rcsbapi

Usage:
    # First pass — downloads data, builds all tiers, fetches sequences,
    # writes MMseqs2 script
    python curate_ppi_dataset.py --outdir ./ppi_dataset --n_pairs 50000

    # Run MMseqs2
    bash ./ppi_dataset/run_mmseqs2.sh

    # Second pass — applies sequence filters, samples final sets
    python curate_ppi_dataset.py --outdir ./ppi_dataset --n_pairs 50000
"""

import json
import time
import random
import argparse
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from negative_builders import (
    build_negatome_direct,
    build_compartment_exclusion_negatives,
    build_degree_matched_cross,
    build_complex_decoy_negatives,
    fetch_subcellular_locations_with_cache,
    find_candidate_complex_pdb_ids,
    DEFAULT_COMPLEX_PDB_IDS_EXAMPLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

HINT_URL      = "https://hint.yulab.org/download-raw/2024-06/HomoSapiens_binary_hq.txt"
# NEGATOME_URL  = "https://mips.helmholtz-muenchen.de/proj/ppi/negatome/manual.txt"
UNIPROT_URL   = "https://rest.uniprot.org/uniprotkb/{uid}.fasta"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> Path:
    """
    Download a file from url to dest. Skips if already exists.
    Returns dest path.
    """
    if dest.exists():
        log.info(f"  Already downloaded: {dest.name}")
        return dest

    log.info(f"  Downloading {url} ...")
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    log.info(f"  Saved to {dest}")
    return dest


def fetch_uniprot_sequence(uid: str) -> str:
    """
    Fetch canonical sequence for a UniProt accession.
    Tries the direct FASTA endpoint first, then falls back to the
    search API which handles merged/retired/TrEMBL accessions.
    Returns sequence string or empty string on failure.
    """
    try:
        r = requests.get(UNIPROT_URL.format(uid=uid), timeout=10)
        if r.status_code == 200:
            lines = r.text.strip().splitlines()
            seq = "".join(l for l in lines if not l.startswith(">"))
            if seq:
                return seq
    except Exception as e:
        log.debug(f"Direct fetch error {uid}: {e}")

    try:
        search_url = (
            "https://rest.uniprot.org/uniprotkb/search"
            f"?query=accession:{uid}&format=fasta&size=1"
        )
        r = requests.get(search_url, timeout=10)
        if r.status_code == 200 and r.text.strip():
            lines = r.text.strip().splitlines()
            seq = "".join(l for l in lines if not l.startswith(">"))
            if seq:
                log.debug(f"  {uid}: resolved via search API (merged/secondary accession)")
                return seq
    except Exception as e:
        log.debug(f"Search API error {uid}: {e}")

    log.debug(f"UniProt fetch failed for {uid} — both endpoints returned no sequence")
    return ""


def fetch_sequences_with_cache(uniprot_ids: list[str],
                                cache_path: Path,
                                refresh: bool = False) -> dict[str, str]:
    """
    Fetch UniProt sequences for a list of IDs with disk caching.
    Returns dict of {uniprot_id: sequence}.
    Already-cached IDs are loaded instantly, only new ones are fetched.
    """
    cache = {}

    if cache_path.exists() and not refresh:
        with open(cache_path) as f:
            cache = json.load(f)
        filled = sum(1 for v in cache.values() if v)
        empty  = sum(1 for v in cache.values() if not v)
        log.info(f"  Loaded sequence cache: {filled} filled, {empty} empty (will retry empty)")

    missing = [uid for uid in uniprot_ids if not cache.get(uid)]

    if missing:
        log.info(f"  Fetching {len(missing)} sequences from UniProt...")
        for uid in tqdm(missing, desc="  UniProt fetch"):
            seq = fetch_uniprot_sequence(uid)
            cache[uid] = seq
            time.sleep(0.05)

        with open(cache_path, "w") as f:
            json.dump(cache, f)
        log.info(f"  Sequence cache updated: {len(cache)} total entries")
    else:
        log.info("  All sequences already cached — skipping UniProt fetch")

    return cache


# ---------------------------------------------------------------------------
# HINT parser
# ---------------------------------------------------------------------------

def load_hint(hint_path: Path) -> list[dict]:
    """
    Parse HINT binary HQ file.
    Format: UniprotA  UniprotB  GeneA  GeneB  [pmids]  [methods]
    Returns list of dicts with keys:
        uniprot_a, uniprot_b  — sorted canonical pair
        evidence_count        — number of times this pair appeared in HINT
        gene_a, gene_b        — gene names where available
    Pairs are deduplicated; evidence_count > 1 indicates multiply-evidenced pairs.
    """
    log.info("Parsing HINT interactions...")
    from collections import Counter

    pair_counts  = Counter()
    pair_genes   = {}
    skipped = 0

    with open(hint_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("uniprot"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                skipped += 1
                continue

            uid_a  = cols[0].strip()
            uid_b  = cols[1].strip()
            gene_a = cols[2].strip() if len(cols) > 2 else ""
            gene_b = cols[3].strip() if len(cols) > 3 else ""

            if not uid_a or not uid_b or uid_a == uid_b:
                skipped += 1
                continue

            if uid_a > uid_b:
                uid_a, uid_b = uid_b, uid_a
                gene_a, gene_b = gene_b, gene_a

            pair_key = (uid_a, uid_b)
            pair_counts[pair_key] += 1
            if pair_key not in pair_genes:
                pair_genes[pair_key] = (gene_a, gene_b)

    pairs = []
    for (uid_a, uid_b), count in pair_counts.most_common():
        gene_a, gene_b = pair_genes[(uid_a, uid_b)]
        pairs.append({
            "uniprot_a":      uid_a,
            "uniprot_b":      uid_b,
            "gene_a":         gene_a,
            "gene_b":         gene_b,
            "evidence_count": count
        })

    multi_evidence = sum(1 for p in pairs if p["evidence_count"] > 1)
    log.info(
        f"  {len(pairs)} unique pairs from {sum(pair_counts.values())} raw interactions "
        f"({skipped} skipped)"
    )
    log.info(
        f"  {multi_evidence} pairs have evidence_count > 1 "
        f"(multiply-evidenced, higher confidence)"
    )
    return pairs


# ---------------------------------------------------------------------------
# Negatome parser
# ---------------------------------------------------------------------------

def load_negatome(negatome_path: Path) -> list[tuple[str, str]]:
    """
    Parse Negatome manual file.
    Format: UniProtID_A  UniProtID_B  PubMedID  Method
    Returns list of (uniprot_a, uniprot_b) pairs.
    """
    log.info("Parsing Negatome pairs...")
    pairs = []
    with open(negatome_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) >= 2:
                pairs.append((cols[0].strip(), cols[1].strip()))

    log.info(f"  {len(pairs)} Negatome pairs loaded")
    return pairs


# ---------------------------------------------------------------------------
# Sequence filters
# ---------------------------------------------------------------------------

def filter_by_length(pairs: list[dict],
                     seq_cache: dict[str, str],
                     max_combined: int = 700) -> list[dict]:
    """
    Remove pairs where combined sequence length > max_combined.
    Also removes pairs where either sequence is missing.
    Mutates each surviving pair in-place to add seq_a/seq_b.
    """
    log.info(f"Filtering by combined sequence length <= {max_combined} aa...")
    filtered = []
    n_missing = 0
    n_toolong = 0

    for pair in pairs:
        seq_a = seq_cache.get(pair["uniprot_a"], "")
        seq_b = seq_cache.get(pair["uniprot_b"], "")

        if not seq_a or not seq_b:
            n_missing += 1
            continue

        combined = len(seq_a) + len(seq_b)
        if combined > max_combined:
            n_toolong += 1
            continue

        pair["seq_a"] = seq_a
        pair["seq_b"] = seq_b
        filtered.append(pair)

    log.info(
        f"  {len(filtered)} pairs kept "
        f"({n_missing} missing sequences, {n_toolong} too long)"
    )
    return filtered


def extract_first_sequence_from_a3m(a3m_path: Path) -> str:
    """
    Extract the query sequence from the first record in an .a3m file.
    Lowercase insertion columns and gap characters are removed.
    """
    seq_lines = []
    in_first = False

    with open(a3m_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if in_first:
                    break
                in_first = True
                continue
            if in_first:
                seq_lines.append(line)

    seq = "".join(seq_lines)
    seq = "".join(c for c in seq if c.isupper() or c == "-")
    return seq.replace("-", "")


def build_msa_assignment_index(msa_dir: Path) -> tuple[set[str], dict[str, str], int]:
    """
    Build indexes from cached a3m files.

    Returns:
      - set of accessions that have a direct cached MSA
      - sequence -> representative accession with a cached MSA
      - count of a3m files whose first sequence could not be parsed
    """
    available_accessions: set[str] = set()
    sequence_to_accession: dict[str, str] = {}
    parse_failures = 0

    if not msa_dir.exists():
        return available_accessions, sequence_to_accession, parse_failures

    for msa_path in sorted(msa_dir.glob("*.a3m")):
        accession = msa_path.stem
        available_accessions.add(accession)
        seq = extract_first_sequence_from_a3m(msa_path)
        if not seq:
            parse_failures += 1
            continue
        sequence_to_accession.setdefault(seq, accession)

    return available_accessions, sequence_to_accession, parse_failures


def resolve_msa_accession(accession: str,
                          sequence: str,
                          available_accessions: set[str],
                          sequence_to_accession: dict[str, str]) -> tuple[str, str]:
    """
    Resolve which accession's MSA should be used for this protein.
    """
    if accession in available_accessions:
        return accession, "direct"

    seq = (sequence or "").strip().upper()
    if seq:
        mapped_accession = sequence_to_accession.get(seq, "")
        if mapped_accession:
            return mapped_accession, "sequence_matched"

    return "", "missing"


# ---------------------------------------------------------------------------
# FASTA writers
# ---------------------------------------------------------------------------

def write_fasta(pairs: list[dict], outpath: Path, id_prefix: str = ""):
    """
    Write sequences from pairs to FASTA for MMseqs2.
    Header format: >{prefix}{uniprot_id}_A or _B
    """
    written = 0
    with open(outpath, "w") as f:
        for pair in pairs:
            uid_a = pair["uniprot_a"]
            uid_b = pair["uniprot_b"]
            f.write(f">{id_prefix}{uid_a}_A\n{pair['seq_a']}\n")
            f.write(f">{id_prefix}{uid_b}_B\n{pair['seq_b']}\n")
            written += 1
    log.info(f"  Written {written} pairs ({written*2} sequences) to {outpath.name}")


# ---------------------------------------------------------------------------
# MMseqs2 script writer
# ---------------------------------------------------------------------------

def write_mmseqs2_script(pos_fasta: Path,
                          neg_fasta: Path,
                          outdir: Path,
                          cluster_identity: float = 0.9,
                          cross_identity: float = 0.3,
                          skip_cross_filter: bool = False) -> Path:
    """
    Write a shell script that runs MMseqs2 within-set clustering (always),
    and optionally a cross-search of positives vs. negatives.

      1. Within-positive clustering  — removes redundant positive pairs
      2. Within-negative clustering  — removes redundant negative pairs
      3. Cross-search pos vs neg     — removes positives too similar to
                                        negatives (skipped if skip_cross_filter)

    cluster_identity and cross_identity are deliberately separate: they
    answer different questions. Clustering (1-2) should use a STRICT
    threshold (default 0.9) -- its job is catching near-identical
    sequences (isoforms, near-duplicate paralogs), not distant homology.
    The previous default of 0.3 here was a bug: at that threshold MMseqs2
    clusters together many proteins that only share a weak, distant motif,
    collapsing the effective number of distinct protein clusters far below
    the true number of unique proteins -- which, combined with the
    apply_cluster_filter bug fixed alongside this, caused a catastrophic
    dataset collapse (see conversation). Cross-filtering (3) answers a
    different question -- "is this positive too similar to some negative,
    such that sequence identity could act as a shortcut" -- and a looser
    threshold (0.3) is appropriate there. Keep them decoupled going forward.

    When skip_cross_filter=True, cross_search.tsv is never generated, so
    apply_cross_filter() in main() naturally no-ops (cross_filter_available
    stays False) without touching clustering dedup, which still runs.
    """
    script       = outdir / "run_mmseqs2.sh"
    tmp          = outdir / "mmseqs2_tmp"
    pos_db       = outdir / "pos_db"
    neg_db       = outdir / "neg_db"
    pos_clust    = outdir / "pos_clustdb"
    neg_clust    = outdir / "neg_clustdb"
    cross_result = outdir / "cross_result"
    pos_tsv      = outdir / "clusters_pos.tsv"
    neg_tsv      = outdir / "clusters_neg.tsv"
    cross_tsv    = outdir / "cross_search.tsv"

    n_steps = 2 if skip_cross_filter else 3

    with open(script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n\n")
        f.write("mkdir -p " + str(tmp) + "\n\n")

        f.write(f"echo '[1/{n_steps}] Building sequence databases...'\n")
        f.write(f"mmseqs createdb {pos_fasta} {pos_db}\n")
        f.write(f"mmseqs createdb {neg_fasta} {neg_db}\n\n")

        f.write(f"echo '[2/{n_steps}] Within-set clustering (min-seq-id={cluster_identity})...'\n")
        f.write(f"mmseqs cluster {pos_db} {pos_clust} {tmp} "
                f"--min-seq-id {cluster_identity} -c 0.8 --cov-mode 0 --threads 8\n")
        f.write(f"mmseqs createtsv {pos_db} {pos_db} {pos_clust} {pos_tsv}\n")
        f.write(f"echo '  Positive clusters written to {pos_tsv.name}'\n\n")

        f.write(f"mmseqs cluster {neg_db} {neg_clust} {tmp} "
                f"--min-seq-id {cluster_identity} -c 0.8 --cov-mode 0 --threads 8\n")
        f.write(f"mmseqs createtsv {neg_db} {neg_db} {neg_clust} {neg_tsv}\n")
        f.write(f"echo '  Negative clusters written to {neg_tsv.name}'\n\n")

        if not skip_cross_filter:
            f.write(f"echo '[3/{n_steps}] Cross-searching positives vs negatives (min-seq-id={cross_identity})...'\n")
            f.write(f"mmseqs search {pos_db} {neg_db} {cross_result} {tmp} "
                    f"--min-seq-id {cross_identity} -c 0.8 --cov-mode 0 "
                    f"-s 7.5 --threads 8\n")
            f.write(f"mmseqs convertalis {pos_db} {neg_db} {cross_result} {cross_tsv} "
                    f"--format-output 'query,target,fident,alnlen,qlen,tlen'\n")
            f.write(f"echo '  Cross-search written to {cross_tsv.name}'\n\n")
        else:
            f.write("echo 'Skipping cross-search by design (--skip_cross_filter) "
                    "-- positives will not be pruned by similarity to negatives.'\n\n")

        f.write("echo 'MMseqs2 filtering complete.'\n")
        f.write("echo 'Re-run curate_ppi_dataset.py to apply filters and sample final sets.'\n")

    script.chmod(0o755)
    log.info(f"  MMseqs2 script written to {script.name}")
    return script




# ---------------------------------------------------------------------------
# Cluster filters
# ---------------------------------------------------------------------------

def apply_cluster_filter(pairs: list[dict],
                          cluster_tsv: Path,
                          id_prefix: str = "") -> list[dict]:
    """
    Remove redundant PAIRS -- two pairs whose chains both fall into the same
    cluster combination (e.g. an isoform swap, or a near-identical paralog
    pair) are collapsed to one. This is a pair-level dedup, not a
    protein-level lockout.

    PREVIOUS VERSION'S BUG (fixed here): the original filter marked each
    individual chain's cluster as "seen" and refused to reuse it in ANY
    later pair, for the rest of the run. That's fine for a protein that
    should only ever appear once, but real PPI networks are full of hub
    proteins that legitimately participate in many true (and many sampled
    negative) pairs -- the old logic capped the whole dataset at roughly
    (unique clusters / 2) pairs, regardless of how well-separated the
    clusters were. This version only rejects a pair if the exact same
    cluster-representative COMBINATION has already been kept, so a hub
    protein can still appear across many different pairs with different
    partners.
    """
    df = pd.read_csv(cluster_tsv, sep="\t", header=None,
                     names=["rep", "member"])
    seq_to_rep = dict(zip(df["member"], df["rep"]))

    seen_pair_keys = set()
    filtered = []

    for pair in pairs:
        id_a = f"{id_prefix}{pair['uniprot_a']}_A"
        id_b = f"{id_prefix}{pair['uniprot_b']}_B"
        rep_a = seq_to_rep.get(id_a, id_a)
        rep_b = seq_to_rep.get(id_b, id_b)

        pair_key = tuple(sorted([rep_a, rep_b]))
        if pair_key in seen_pair_keys:
            continue
        seen_pair_keys.add(pair_key)
        filtered.append(pair)

    log.info(f"  Cluster filter: {len(pairs)} → {len(filtered)} pairs")
    return filtered


def apply_cross_filter(positive_pairs: list[dict],
                        cross_tsv: Path,
                        identity_threshold: float = 0.3) -> list[dict]:
    """
    Remove positive pairs where either chain hits a negative sequence
    above identity_threshold in the MMseqs2 cross-search.
    """
    df = pd.read_csv(
        cross_tsv, sep="\t", header=None,
        names=["query", "target", "fident", "alnlen", "qlen", "tlen"]
    )
    flagged = set(df[df["fident"] >= identity_threshold]["query"].unique())

    filtered = []
    removed = 0
    for pair in positive_pairs:
        id_a = f"{pair['uniprot_a']}_A"
        id_b = f"{pair['uniprot_b']}_B"
        if id_a in flagged or id_b in flagged:
            removed += 1
        else:
            filtered.append(pair)

    log.info(f"  Cross-filter: {len(positive_pairs)} → {len(filtered)} pairs "
             f"({removed} removed)")
    return filtered


# ---------------------------------------------------------------------------
# Tiered negative construction (orchestrates negative_builders.py)
# ---------------------------------------------------------------------------

def build_all_negative_tiers(negatome_pairs, positive_pairs, hint_set,
                              outdir: Path, args) -> list[dict]:
    """
    Build negatives from all four tiers and combine them, logging a
    per-tier breakdown. `hint_set` should be the FULL (unfiltered) set of
    HINT pairs -- see module docstring for why this matters.
    """
    all_negatives = []

    # --- Tier 1: direct Negatome pairs (all of them; small, keep everything) ---
    negatome_direct = build_negatome_direct(negatome_pairs, hint_set)
    all_negatives.extend(negatome_direct)

    # --- Tier 2: complex decoys (small, precious hard negatives -- added on
    #     top of the bulk-volume target rather than counted against it) ---
    complex_decoys = []
    if not args.skip_complex_decoy:
        complex_pdb_ids = DEFAULT_COMPLEX_PDB_IDS_EXAMPLE
        if args.complex_pdb_ids:
            complex_pdb_ids = [x.strip() for x in args.complex_pdb_ids.split(",") if x.strip()]
        elif args.auto_find_complexes:
            complex_pdb_ids = find_candidate_complex_pdb_ids(
                min_polymer_entities=5, max_results=args.max_complexes
            )
        else:
            log.warning(
                "  No --complex_pdb_ids given and --auto_find_complexes not set — "
                f"using a tiny example list ({complex_pdb_ids}). Verify these IDs "
                "or pass your own before relying on this tier."
            )
        complex_decoys = build_complex_decoy_negatives(
            complex_pdb_ids, hint_set,
            cache_dir=outdir / "pdb_cache"
        )
        all_negatives.extend(complex_decoys)
    else:
        log.info("  Skipping complex_decoy tier (--skip_complex_decoy)")

    # --- Remaining bulk volume, split between compartment_exclusion (primary)
    #     and degree_matched_cross (fills whatever compartment_exclusion can't) ---
    n_bulk_target = max(0, args.n_pairs * 3 - len(negatome_direct) - len(complex_decoys))
    n_compartment_target = int(n_bulk_target * args.compartment_fraction)
    n_cross_target = n_bulk_target - n_compartment_target

    compartment_negs = []
    if not args.skip_compartment and n_compartment_target > 0:
        candidate_proteins = sorted({p for a, b in negatome_pairs for p in (a, b)}
                                     | {pp["uniprot_a"] for pp in positive_pairs}
                                     | {pp["uniprot_b"] for pp in positive_pairs})
        loc_cache_path = outdir / "subcellular_location_cache.json"
        loc_cache = fetch_subcellular_locations_with_cache(
            candidate_proteins, loc_cache_path, refresh=args.refresh
        )
        compartment_negs = build_compartment_exclusion_negatives(
            candidate_proteins, loc_cache, hint_set, n_compartment_target
        )
        all_negatives.extend(compartment_negs)
    else:
        log.info("  Skipping compartment_exclusion tier")

    # Recompute remaining target in case compartment tier came up short
    n_shortfall = n_compartment_target - len(compartment_negs)
    n_cross_target += max(0, n_shortfall)

    if not args.skip_degree_cross and n_cross_target > 0:
        cross_negs = build_degree_matched_cross(
            negatome_pairs, positive_pairs, hint_set, n_cross_target
        )
        all_negatives.extend(cross_negs)
    else:
        log.info("  Skipping degree_matched_cross tier")

    tier_counts = pd.Series([n["neg_source"] for n in all_negatives]).value_counts()
    log.info("  Negative tier breakdown:\n" + tier_counts.to_string())

    return all_negatives


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)

    # ----------------------------------------------------------------
    # Step 1: Download HINT and Negatome
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 1: Downloading source databases")
    log.info("=" * 60)

    hint_path     = outdir / "HomoSapiens_binary_hq.txt"
    negatome_path = outdir / "negatome_ms.txt"

    download_file(HINT_URL, hint_path)
    # download_file(NEGATOME_URL, negatome_path)  # place negatome_ms.txt manually

    # ----------------------------------------------------------------
    # Step 2: Parse interactions
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 2: Parsing interactions")
    log.info("=" * 60)

    hint_pairs     = load_hint(hint_path)
    negatome_pairs = load_negatome(negatome_path)

    # Full HINT pair set -- used for exclusion across ALL negative tiers.
    # (Previously computed but never actually passed into negative
    # construction -- see module docstring.)
    hint_set = set(tuple(sorted([p["uniprot_a"], p["uniprot_b"]])) for p in hint_pairs)

    # ----------------------------------------------------------------
    # Step 3: Build positive pairs (metadata only -- sequences fetched below)
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 3: Building positive set (pre-sequence)")
    log.info("=" * 60)

    positive_raw = [
        {
            "uniprot_a":      p["uniprot_a"],
            "uniprot_b":      p["uniprot_b"],
            "gene_a":         p.get("gene_a", ""),
            "gene_b":         p.get("gene_b", ""),
            "evidence_count": p.get("evidence_count", 1),
            "label":          1
        }
        for p in hint_pairs
    ]

    # ----------------------------------------------------------------
    # Step 4: Build all negative tiers (metadata only -- may introduce new
    # UniProt IDs, e.g. complex_decoy proteins never seen in HINT/Negatome)
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 4: Building negative tiers")
    log.info("=" * 60)

    negatives_raw = build_all_negative_tiers(
        negatome_pairs, positive_raw, hint_set, outdir, args
    )

    # ----------------------------------------------------------------
    # Step 5: Fetch sequences for the UNION of all proteins across every
    # tier (positives + all four negative tiers), then apply length filter
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 5: Fetching UniProt sequences (union across all tiers)")
    log.info("=" * 60)

    all_uids = set()
    for p in positive_raw:
        all_uids.add(p["uniprot_a"]); all_uids.add(p["uniprot_b"])
    for n in negatives_raw:
        all_uids.add(n["uniprot_a"]); all_uids.add(n["uniprot_b"])

    log.info(f"  Total unique UniProt IDs to fetch: {len(all_uids)}")

    seq_cache_path = outdir / "sequence_cache.json"
    seq_cache = fetch_sequences_with_cache(
        list(all_uids), seq_cache_path, refresh=args.refresh
    )

    log.info(f"  Applying combined length filter (<= {args.max_len} aa) to positives...")
    positives = filter_by_length(positive_raw, seq_cache, max_combined=args.max_len)

    log.info(f"  Applying combined length filter (<= {args.max_len} aa) to negatives...")
    negatives_raw = filter_by_length(negatives_raw, seq_cache, max_combined=args.max_len)

    tier_counts_post = pd.Series([n["neg_source"] for n in negatives_raw]).value_counts()
    log.info("  Negative tier breakdown after sequence/length filtering:\n"
              + tier_counts_post.to_string())

    # ----------------------------------------------------------------
    # Step 6: Write FASTAs and MMseqs2 script
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 6: Writing FASTAs and MMseqs2 script")
    log.info("=" * 60)

    pos_fasta = outdir / "positives.fasta"
    neg_fasta = outdir / "negatives.fasta"

    write_fasta(positives, pos_fasta)
    write_fasta(negatives_raw, neg_fasta)

    mmseqs_script = write_mmseqs2_script(
        pos_fasta, neg_fasta, outdir,
        cluster_identity=args.cluster_identity,
        cross_identity=args.seq_identity,
        skip_cross_filter=args.skip_cross_filter,
    )

    # ----------------------------------------------------------------
    # Step 7: Apply MMseqs2 filters if available
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 7: Applying sequence similarity filters")
    log.info("=" * 60)

    pos_cluster_tsv = outdir / "clusters_pos.tsv"
    neg_cluster_tsv = outdir / "clusters_neg.tsv"
    cross_tsv       = outdir / "cross_search.tsv"

    # Split on purpose: within-set dedup (clustering) and positive-pruning-by-
    # similarity-to-negatives (cross-filter) are independent steps. Bundling
    # them into one flag meant disabling cross-search (e.g. by dropping part 3
    # from run_mmseqs2.sh) silently disabled clustering dedup too.
    cluster_filters_available = pos_cluster_tsv.exists() and neg_cluster_tsv.exists()
    cross_filter_available    = cross_tsv.exists()
    mmseqs_available = cluster_filters_available or cross_filter_available  # for the summary log below

    if cluster_filters_available:
        log.info("  Cluster outputs found — applying within-set dedup")
        positives = apply_cluster_filter(positives, pos_cluster_tsv)
        negatives_raw = apply_cluster_filter(negatives_raw, neg_cluster_tsv)
    else:
        log.warning(
            "  Cluster outputs not found — within-set dedup not applied.\n"
            f"  Run: bash {mmseqs_script}\n"
            "  Then re-run this script to apply filters."
        )

    if cross_filter_available:
        log.info("  Cross-search output found — pruning positives similar to negatives")
        positives = apply_cross_filter(positives, cross_tsv, identity_threshold=args.seq_identity)
    else:
        log.info("  Skipping cross-filter (no cross_search.tsv) — positives left unpruned by design")

    # ----------------------------------------------------------------
    # Step 8: Sample to target size (stratified across neg_source so no
    # single tier is accidentally dropped entirely by random sampling)
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 8: Sampling final balanced dataset")
    log.info("=" * 60)

    n_pos = min(len(positives), args.n_pairs)
    n_neg = min(len(negatives_raw), args.n_pairs)
    n_final = min(n_pos, n_neg)

    if n_final < args.n_pairs:
        log.warning(
            f"  Only {n_final} pairs available per class "
            f"(target was {args.n_pairs}). Consider relaxing filters, "
            f"increasing n_pairs, or expanding a tier's target size."
        )

    pos_idx = rng.choice(len(positives), size=n_final, replace=False)
    final_positives = [positives[i] for i in pos_idx]

    # Stratified sampling: keep each tier's share of the final negative set
    # proportional to its share of the filtered pool, rather than one flat
    # random draw that could (by chance) starve a small tier like
    # negatome_direct.
    neg_by_tier = {}
    for n in negatives_raw:
        neg_by_tier.setdefault(n["neg_source"], []).append(n)

    final_negatives = []
    for tier, tier_pairs in neg_by_tier.items():
        tier_share = round(n_final * len(tier_pairs) / len(negatives_raw))
        tier_share = min(tier_share, len(tier_pairs))
        idx = rng.choice(len(tier_pairs), size=tier_share, replace=False)
        final_negatives.extend([tier_pairs[i] for i in idx])
    # Trim/pad to exactly n_final if rounding drifted
    if len(final_negatives) > n_final:
        final_negatives = list(rng.choice(final_negatives, size=n_final, replace=False))
    elif len(final_negatives) < n_final:
        remaining_pool = [n for n in negatives_raw if n not in final_negatives]
        top_up = rng.choice(len(remaining_pool), size=n_final - len(final_negatives), replace=False)
        final_negatives.extend([remaining_pool[i] for i in top_up])

    log.info(f"  Final positives: {len(final_positives)}")
    log.info(f"  Final negatives: {len(final_negatives)}")
    final_tier_counts = pd.Series([n["neg_source"] for n in final_negatives]).value_counts()
    log.info("  Final negative tier breakdown:\n" + final_tier_counts.to_string())

    # ----------------------------------------------------------------
    # Step 9: Save outputs
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 9: Saving outputs")
    log.info("=" * 60)

    pos_out = outdir / "positives.json"
    neg_out = outdir / "negatives.json"

    with open(pos_out, "w") as f:
        json.dump(final_positives, f, indent=2)
    log.info(f"  Positives saved: {pos_out}")

    with open(neg_out, "w") as f:
        json.dump(final_negatives, f, indent=2)
    log.info(f"  Negatives saved: {neg_out}")

    all_pairs = (
        [{**p, "label": 1, "neg_source": "positive"} for p in final_positives] +
        [{**n, "label": 0} for n in final_negatives]
    )

    msa_dir = Path(args.msa_dir) if args.msa_dir else outdir / "inputs" / "msa_cache"
    available_msa_accessions, seq_to_msa_accession, msa_parse_failures = build_msa_assignment_index(msa_dir)
    log.info(
        "  MSA assignment index: %d direct accessions, %d unique query sequences from %s",
        len(available_msa_accessions),
        len(seq_to_msa_accession),
        msa_dir,
    )
    if msa_parse_failures:
        log.warning("  %d a3m file(s) had no parseable query sequence", msa_parse_failures)

    assigned_map: dict[str, tuple[str, str]] = {}
    summary_rows = []
    for p in all_pairs:
        msa_acc_a, msa_source_a = resolve_msa_accession(
            p["uniprot_a"],
            p["seq_a"],
            available_msa_accessions,
            seq_to_msa_accession,
        )
        msa_acc_b, msa_source_b = resolve_msa_accession(
            p["uniprot_b"],
            p["seq_b"],
            available_msa_accessions,
            seq_to_msa_accession,
        )

        if msa_source_a == "sequence_matched" and p["uniprot_a"] != msa_acc_a:
            assigned_map.setdefault(p["uniprot_a"], (msa_acc_a, msa_source_a))
        if msa_source_b == "sequence_matched" and p["uniprot_b"] != msa_acc_b:
            assigned_map.setdefault(p["uniprot_b"], (msa_acc_b, msa_source_b))

        summary_rows.append({
            "uniprot_a":      p["uniprot_a"],
            "uniprot_b":      p["uniprot_b"],
            "gene_a":         p.get("gene_a", ""),
            "gene_b":         p.get("gene_b", ""),
            "len_a":          len(p["seq_a"]),
            "len_b":          len(p["seq_b"]),
            "combined":       len(p["seq_a"]) + len(p["seq_b"]),
            "evidence_count": p.get("evidence_count", 0),
            "label":          p["label"],
            "neg_source":     p["neg_source"],
            "msa_acc_a":      msa_acc_a,
            "msa_acc_b":      msa_acc_b,
            "msa_source_a":   msa_source_a,
            "msa_source_b":   msa_source_b,
        })

    df = pd.DataFrame(summary_rows)
    summary_path = outdir / "dataset_summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)
    log.info(f"  Summary saved: {summary_path}")

    assigned_path = outdir / "assigned_msas.txt"
    with open(assigned_path, "w") as f:
        f.write("accession\tmapped_to_msa\tsource\n")
        for accession, (mapped_to, source) in sorted(assigned_map.items()):
            f.write(f"{accession}\t{mapped_to}.a3m\t{source}\n")
    log.info(f"  Assigned MSA mappings: {len(assigned_map)} accessions -> {assigned_path}")

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Total pairs       : {len(all_pairs)}")
    log.info(f"  Positives         : {len(final_positives)}")
    log.info(f"  Negatives         : {len(final_negatives)}")
    log.info(f"  Output directory  : {outdir}")
    if not mmseqs_available:
        log.info("")
        log.info("  Next steps:")
        log.info(f"    1. bash {mmseqs_script}")
        log.info(f"    2. python curate_ppi_dataset.py --outdir {outdir} --n_pairs {args.n_pairs}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate balanced PPI dataset from HINT + Negatome + compartment + complex-decoy negatives"
    )
    parser.add_argument("--outdir",       default="./ppi_dataset")
    parser.add_argument("--n_pairs",      type=int,   default=50000,
                        help="Target number of pairs per class (final, post-filtering)")
    parser.add_argument("--max_len",      type=int,   default=2000)
    parser.add_argument("--seq_identity", type=float, default=0.3,
                        help="Identity threshold for the cross-filter (positives-vs-negatives "
                             "similarity pruning), if not skipped. Not used for clustering.")
    parser.add_argument("--cluster_identity", type=float, default=0.9,
                        help="Identity threshold for within-set clustering (redundancy removal). "
                             "Should be strict (near-duplicate/isoform level) -- this used to "
                             "default to 0.3 (shared with cross-filter), which was too permissive "
                             "and, combined with a pair-vs-protein-level filtering bug, collapsed "
                             "datasets by ~95%%. See conversation for details.")
    parser.add_argument("--refresh",      action="store_true")
    parser.add_argument("--msa_dir",      type=str, default="",
                        help="Directory containing cached per-accession .a3m files "
                            "used to resolve msa_acc_a/msa_acc_b in dataset_summary.tsv. "
                            "Default: <outdir>/inputs/msa_cache")
    parser.add_argument("--skip_cross_filter", action="store_true",
                        help="Don't run the MMseqs2 cross-search step, so positives "
                             "are never pruned for sequence similarity to negatives. "
                             "Within-set clustering dedup still runs.")

    # Tier controls
    parser.add_argument("--compartment_fraction", type=float, default=0.6,
                        help="Fraction of remaining bulk negative volume "
                             "(after negatome_direct + complex_decoy) to fill "
                             "with compartment_exclusion vs. degree_matched_cross")
    parser.add_argument("--skip_compartment",   action="store_true")
    parser.add_argument("--skip_degree_cross",  action="store_true")
    parser.add_argument("--skip_complex_decoy", action="store_true")
    parser.add_argument("--complex_pdb_ids", type=str, default=None,
                        help="Comma-separated PDB IDs for the complex_decoy tier. "
                             "If omitted, uses --auto_find_complexes or a tiny example list.")
    parser.add_argument("--auto_find_complexes", action="store_true",
                        help="Use find_candidate_complex_pdb_ids() (rcsbapi search) "
                             "instead of a hardcoded/example PDB ID list.")
    parser.add_argument("--max_complexes", type=int, default=5000)

    args = parser.parse_args()
    main(args)