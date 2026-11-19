#!/usr/bin/env python3
"""
Pool AF3 pair and single embeddings and append to HDF5 dataset.
================================================================
Processes one AF3 output directory at a time — designed to be called
from a SLURM job script after each AF3 run completes.

Reads:
  {af3_out_dir}/seed-{seed}_embeddings/*.npz (default seed: 1)
    keys: pair_embeddings   [N, N, 128]
          single_embeddings [N, 384]

Pools cross-chain submatrix [L_A, L_B, 128] and stores each component
as a separate HDF5 key for full ablation flexibility at training time:

  /pair_row_mean    [N, 128]  — mean over L_B → mean over L_A
  /pair_row_max     [N, 128]  — mean over L_B → max  over L_A
  /pair_col_mean    [N, 128]  — mean over L_A → mean over L_B
  /pair_col_max     [N, 128]  — mean over L_A → max  over L_B
  /single_a_mean    [N, 384]  — mean over L_A residues
  /single_a_max     [N, 384]  — max  over L_A residues
  /single_b_mean    [N, 384]  — mean over L_B residues
  /single_b_max     [N, 384]  — max  over L_B residues
  /system_name      [N]       string
  /uniprot_a        [N]       string
  /uniprot_b        [N]       string
  /neg_source       [N]       string  ("" for positives, tier name for
                                        negatives -- negatome_direct /
                                        compartment_exclusion /
                                        degree_matched_cross /
                                        complex_decoy)
  /label            [N]       int8    (1=positive, 0=negative)
  /evidence_count   [N]       int16

At training time concatenate whichever components you want, e.g.:
  full  = np.concatenate([pair_row_mean, pair_row_max,
                          pair_col_mean, pair_col_max,
                          single_a_mean, single_a_max,
                          single_b_mean, single_b_max], axis=-1)  # [2048]
  pair  = np.concatenate([pair_row_mean, pair_row_max,
                          pair_col_mean, pair_col_max], axis=-1)  # [512]

Deletion behaviour (default: delete embeddings subfolder only):
  default        — delete seed-1_embeddings/ after pooling
  --delete-all   — delete entire af3_out_dir after pooling
  --keep-npz     — keep everything (overrides --delete-all)

Usage:
    # Latent screen — delete full output folder
    python pool_embeddings.py \\
        --af3-out-dir ./af3_predictions/P04637_Q9Y4K3 \\
        --summary-tsv ./dataset_summary.tsv \\
        --delete-all

    # Complex interp — keep structures, delete embeddings only
    python pool_embeddings.py \\
        --af3-out-dir ./af3_predictions/P04637_Q9Y4K3 \\
        --summary-tsv ./dataset_summary.tsv

    # Debug — keep everything
    python pool_embeddings.py \\
        --af3-out-dir ./af3_predictions/P04637_Q9Y4K3 \\
        --summary-tsv ./dataset_summary.tsv \\
        --keep-npz
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

PAIR_DIM   = 128
SINGLE_DIM = 384

# HDF5 dataset names and their shapes — used for init and append
EMBEDDING_KEYS = {
    "pair_row_mean": PAIR_DIM,
    "pair_row_max":  PAIR_DIM,
    "pair_col_mean": PAIR_DIM,
    "pair_col_max":  PAIR_DIM,
    "single_a_mean": SINGLE_DIM,
    "single_a_max":  SINGLE_DIM,
    "single_b_mean": SINGLE_DIM,
    "single_b_max":  SINGLE_DIM,
}

# EDITED: added "neg_source" -- string metadata field, empty for
# positives, tier name for negatives. Handled generically by the same
# init/resize loops as system_name/uniprot_a/uniprot_b below.
METADATA_KEYS = ["system_name", "uniprot_a", "uniprot_b", "neg_source"]


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool_pair(pair_emb: np.ndarray,
              len_a: int,
              len_b: int) -> dict[str, np.ndarray]:
    """
    Extract cross-chain submatrix and apply row/col mean+max pooling.

    Args:
        pair_emb: full pair representation [N, N, 128]
        len_a:    number of residues in chain A
        len_b:    number of residues in chain B

    Returns:
        dict of pooled vectors each [128]
    """
    if len_a <= 0 or len_b <= 0:
        raise ValueError("Both protein chain lengths must be positive")
    expected = (len_a + len_b, len_a + len_b, PAIR_DIM)
    if pair_emb.shape != expected:
        raise ValueError(f"Expected pair embeddings {expected}, got {pair_emb.shape}")

    # Cross-chain submatrix: rows = chain A, cols = chain B
    cross = pair_emb[:len_a, len_a:, :]    # [L_A, L_B, 128]

    # Row-wise: for each residue in A, summarise relationship to all of B
    row_pooled = cross.mean(axis=1, dtype=np.float32)  # [L_A, 128]
    row_mean   = row_pooled.mean(axis=0)    # [128]
    row_max    = row_pooled.max(axis=0)     # [128]

    # Col-wise: for each residue in B, summarise relationship to all of A
    col_pooled = cross.mean(axis=0, dtype=np.float32)  # [L_B, 128]
    col_mean   = col_pooled.mean(axis=0)    # [128]
    col_max    = col_pooled.max(axis=0)     # [128]

    return {
        "pair_row_mean": row_mean,
        "pair_row_max":  row_max,
        "pair_col_mean": col_mean,
        "pair_col_max":  col_max,
    }


def pool_single(single_emb: np.ndarray,
                len_a: int,
                len_b: int) -> dict[str, np.ndarray]:
    """
    Split single representation into chain A and B, apply mean+max.

    Args:
        single_emb: full single representation [N, 384]
        len_a:      number of residues in chain A
        len_b:      number of residues in chain B

    Returns:
        dict of pooled vectors each [384]
    """
    expected = (len_a + len_b, SINGLE_DIM)
    if len_a <= 0 or len_b <= 0 or single_emb.shape != expected:
        raise ValueError(f"Expected positive chain lengths and single embeddings {expected}, got {single_emb.shape}")
    single_a = single_emb[:len_a, :]        # [L_A, 384]
    single_b = single_emb[len_a:, :]        # [L_B, 384]

    return {
        "single_a_mean": single_a.mean(axis=0, dtype=np.float32),
        "single_a_max":  single_a.max(axis=0).astype(np.float32),
        "single_b_mean": single_b.mean(axis=0, dtype=np.float32),
        "single_b_max":  single_b.max(axis=0).astype(np.float32),
    }


def pool_embeddings(npz_path: Path,
                    len_a: int,
                    len_b: int) -> dict[str, np.ndarray]:
    """
    Load NPZ, pool pair and single representations.
    Returns dict of all 8 pooled component vectors.
    """
    # Accumulate means in float32 without copying the entire quadratic pair
    # tensor to float32 (AF3 normally saves float16 arrays).
    with np.load(npz_path, allow_pickle=False) as data:
        components = pool_pair(data["pair_embeddings"], len_a, len_b)
        components.update(pool_single(data["single_embeddings"], len_a, len_b))
    if any(not np.isfinite(vec).all() for vec in components.values()):
        raise ValueError("Pooled embeddings contain NaN or infinity")

    return components


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def init_hdf5(hdf5_path: Path) -> None:
    """
    Initialise an empty HDF5 file with resizable datasets.
    Called only if the file does not exist yet.
    """
    log.info(f"Initialising new HDF5 dataset: {hdf5_path}")
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:

        # Embedding components — one dataset per pooled vector
        for key, dim in EMBEDDING_KEYS.items():
            f.create_dataset(
                key,
                shape=(0, dim),
                maxshape=(None, dim),
                dtype=np.float32,
                chunks=(256, dim),
                compression="gzip",
                shuffle=True,
            )

        # String metadata (system_name, uniprot_a, uniprot_b, neg_source)
        str_dt = h5py.special_dtype(vlen=str)
        for key in METADATA_KEYS:
            f.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=str_dt
            )

        # Integer labels
        f.create_dataset(
            "label",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int8
        )
        f.create_dataset(
            "evidence_count",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int16
        )

        # Document the structure
        f.attrs["description"] = (
            "LatentPPI pooled embeddings. "
            "Each embedding key stores one pooled component per pair. "
            "Concatenate components at training time for desired input vector."
        )
        f.attrs["pair_dim"]   = PAIR_DIM
        f.attrs["single_dim"] = SINGLE_DIM
        f.attrs["full_dim"]   = 4 * PAIR_DIM + 4 * SINGLE_DIM


def _ensure_backward_compat_datasets(f: h5py.File) -> None:
    """
    EDITED: handles HDF5 files created before neg_source was added to
    METADATA_KEYS. Backfills existing rows with empty string so
    appends don't fail on a missing dataset. No-op if neg_source
    already exists (i.e. the file was created with this version).
    """
    if "neg_source" not in f:
        n_existing = f["pair_row_mean"].shape[0]
        str_dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset(
            "neg_source",
            shape=(n_existing,),
            maxshape=(None,),
            dtype=str_dt,
        )
        ds[:] = ""
        log.warning(
            f"  Backfilled 'neg_source' for {n_existing} pre-existing rows "
            f"with empty string (dataset didn't exist in this HDF5 yet)"
        )


def append_to_hdf5(hdf5_path: Path,
                   components: dict[str, np.ndarray],
                   system_name: str,
                   uniprot_a: str,
                   uniprot_b: str,
                   neg_source: str,
                   label: int,
                   evidence_count: int) -> None:
    """
    Append one row to each dataset in the HDF5 file.
    Initialises the file if it does not exist.
    """
    if not hdf5_path.exists():
        init_hdf5(hdf5_path)

    with h5py.File(hdf5_path, "a") as f:
        _ensure_backward_compat_datasets(f)  # EDITED

        n = f["pair_row_mean"].shape[0]

        # Resize all datasets by 1
        for key in list(EMBEDDING_KEYS.keys()) + METADATA_KEYS + ["label", "evidence_count"]:
            f[key].resize(n + 1, axis=0)

        # Write embedding components
        for key in EMBEDDING_KEYS:
            f[key][n] = components[key]

        # Write metadata
        f["system_name"][n]    = system_name
        f["uniprot_a"][n]      = uniprot_a
        f["uniprot_b"][n]      = uniprot_b
        f["neg_source"][n]     = neg_source  # EDITED
        f["label"][n]          = label
        f["evidence_count"][n] = evidence_count

    log.info(
        f"  Appended row {n+1}: {system_name} "
        f"(label={label}, neg_source='{neg_source}', evidence_count={evidence_count})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    af3_out_dir = Path(args.af3_out_dir)
    summary_tsv = Path(args.summary_tsv)
    hdf5_path   = Path(args.hdf5_path)

    # ----------------------------------------------------------------
    # Resolve system name from directory name
    # ----------------------------------------------------------------
    system_name = af3_out_dir.name
    log.info(f"Processing: {system_name}")

    # ----------------------------------------------------------------
    # Find exactly one NPZ for the selected seed; never silently pick a file.
    # ----------------------------------------------------------------
    embed_folder = af3_out_dir / f"seed-{args.seed}_embeddings"
    if args.npz_path:
        npz_path = Path(args.npz_path)
        if not npz_path.is_file():
            log.error(f"Embedding file not found: {npz_path}")
            sys.exit(1)
        if not args.keep_npz:
            log.error("Explicit --npz-path requires --keep-npz; clean up after all stages are pooled")
            sys.exit(1)
    else:
        npz_candidates = list(embed_folder.glob("*.npz"))
        if len(npz_candidates) != 1:
            log.error(f"Expected exactly one .npz in {embed_folder}, found {len(npz_candidates)}")
            sys.exit(1)
        npz_path = npz_candidates[0]
    log.info(f"  NPZ: {npz_path}")

    # ----------------------------------------------------------------
    # Load pair metadata from summary TSV
    # ----------------------------------------------------------------
    if not summary_tsv.exists():
        log.error(f"Summary TSV not found: {summary_tsv}")
        sys.exit(1)

    df = pd.read_csv(summary_tsv, sep="\t", dtype={"uniprot_a": str, "uniprot_b": str})
    df["system_name"] = df["uniprot_a"] + "_" + df["uniprot_b"]
    # AF3 lowercases output names; retain canonical accession case in HDF5.
    row = df[df["system_name"].str.lower() == system_name.lower()]

    if row.empty:
        log.error(
            f"System {system_name} not found in {summary_tsv}. "
            f"Check that uniprot_a_uniprot_b matches the directory name."
        )
        sys.exit(1)

    if len(row) != 1:
        log.error(f"Ambiguous metadata: {len(row)} rows match {system_name}")
        sys.exit(1)
    row            = row.iloc[0]
    system_name    = str(row["system_name"])
    len_a          = int(row["len_a"])
    len_b          = int(row["len_b"])
    uniprot_a      = str(row["uniprot_a"])
    uniprot_b      = str(row["uniprot_b"])
    label          = int(row.get("label", -1))
    evidence_count = int(row.get("evidence_count", 0))

    # EDITED: pull neg_source, defaulting to "" for positives / missing column.
    # row.get(..., "") only applies its default if the column is absent
    # entirely -- if the column exists but the cell is blank, pandas
    # gives NaN, so that's handled explicitly too.
    neg_source_val = row.get("neg_source", "")
    if pd.isna(neg_source_val):
        neg_source_val = ""
    neg_source = str(neg_source_val)

    log.info(f"  Chain lengths: A={len_a}, B={len_b} (combined={len_a+len_b})")
    log.info(f"  label={label}, neg_source='{neg_source}', evidence_count={evidence_count}")

    # ----------------------------------------------------------------
    # Pool embeddings
    # ----------------------------------------------------------------
    log.info("  Pooling embeddings...")
    try:
        components = pool_embeddings(npz_path, len_a, len_b)
    except ValueError as e:
        log.error(f"  Pooling failed — {e}")
        sys.exit(1)
    except KeyError as e:
        log.error(
            f"  Key {e} not found in NPZ. "
            f"Available keys: {list(np.load(npz_path).keys())}"
        )
        sys.exit(1)

    for key, vec in components.items():
        log.info(f"  {key}: {vec.shape}")

    # ----------------------------------------------------------------
    # Append to HDF5
    # ----------------------------------------------------------------
    append_to_hdf5(
        hdf5_path=hdf5_path,
        components=components,
        system_name=system_name,
        uniprot_a=uniprot_a,
        uniprot_b=uniprot_b,
        neg_source=neg_source,  # EDITED
        label=label,
        evidence_count=evidence_count
    )

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    if not args.keep_npz:
        if args.delete_all:
            shutil.rmtree(af3_out_dir)
            log.info(f"  Deleted full AF3 output folder: {af3_out_dir}")
        else:
            shutil.rmtree(embed_folder)
            log.info(f"  Deleted embeddings folder: {embed_folder}")

    log.info(f"  Done: {system_name} → {hdf5_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pool AF3 embeddings and append to HDF5 dataset"
    )
    parser.add_argument(
        "--af3-out-dir",
        required=True,
        help="AF3 output directory for one pair (e.g. af3_predictions/P04637_Q9Y4K3)"
    )
    parser.add_argument(
        "--summary-tsv",
        required=True,
        help="dataset_summary.tsv with uniprot_a, uniprot_b, len_a, len_b, label, neg_source, evidence_count"
    )
    parser.add_argument(
        "--hdf5-path",
        default="embeddings.h5",
        help="Path to HDF5 output file (default: embeddings.h5)"
    )
    parser.add_argument("--npz-path", help="Exact embedding NPZ to pool; requires --keep-npz")
    parser.add_argument(
        "--seed", type=int, default=1,
        help="AF3 seed whose embeddings to pool (default: 1; one seed per shard)"
    )
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete entire AF3 output folder after pooling (default: delete embeddings subfolder only)"
    )
    parser.add_argument(
        "--keep-npz",
        action="store_true",
        help="Keep all AF3 outputs after pooling — overrides --delete-all"
    )

    args = parser.parse_args()
    main(args)
