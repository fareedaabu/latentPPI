# Written by FEA 070826. Parser to add ipsae scores to embeds.py
#!/usr/bin/env python3
"""
Add ipSAE and related metrics to embeddings HDF5.
===================================================
Reads an ipSAE summary CSV, matches systems to the embeddings HDF5
by system_name, and writes all metric columns as new datasets.

For each system, takes the row with the maximum ipSAE value
(handles multiple rows per system from A→B, B→A, and seed repeats).

Systems in HDF5 with no CSV match are skipped with a warning.
Re-running is safe — existing metric datasets are overwritten.

Metric columns added to HDF5:
  plddt, ipSAE, ipSAE_d0chn, ipSAE_d0dom,
  ipTM_af, ipTM_d0chn, pDockQ, pDockQ2, LIS

Usage:
    python add_metrics_to_hdf5.py \\
        --hdf5-path  ./embeddings.h5 \\
        --ipsae-csv  ./ipsae_summary.csv
"""

import argparse
import logging
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

# Metric columns to extract and store
METRIC_COLS = [
    "plddt",
    "ipSAE",
    "ipSAE_d0chn",
    "ipSAE_d0dom",
    "ipTM_af",
    "ipTM_d0chn",
    "pDockQ",
    "pDockQ2",
    "LIS",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ipsae_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load ipSAE CSV and return one row per system — the row with max ipSAE.
    """
    log.info(f"Loading ipSAE CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    log.info(f"  {len(df)} total rows, {df['system'].nunique()} unique systems")

    # Take the row with max ipSAE per system
    df = df.sort_values("ipSAE", ascending=False)
    df = df.drop_duplicates(subset="system", keep="first")

    log.info(f"  {len(df)} systems after taking max ipSAE row per system")
    return df.set_index("system")


def get_hdf5_systems(hdf5_path: Path) -> list[str]:
    """
    Read system_name dataset from HDF5, return as list of strings.
    """
    with h5py.File(hdf5_path, "r") as f:
        systems = [s.decode() if isinstance(s, bytes) else s
                   for s in f["system_name"][:]]
    return systems


def init_metric_datasets(hdf5_path: Path, n: int) -> None:
    """
    Create metric datasets in HDF5 if they don't exist.
    Initialised to NaN so missing systems are distinguishable.
    """
    with h5py.File(hdf5_path, "a") as f:
        for col in METRIC_COLS:
            if col not in f:
                ds = f.create_dataset(
                    col,
                    shape=(n,),
                    maxshape=(None,),
                    dtype=np.float32
                )
                ds[:] = np.nan
                log.info(f"  Initialised dataset: {col} [{n}]")


def write_metrics(hdf5_path: Path,
                  systems: list[str],
                  ipsae_df: pd.DataFrame) -> dict:
    """
    Write metric values into HDF5 for each matched system.
    Returns summary counts.
    """
    n_matched = 0
    n_missing = 0
    missing_log = []

    with h5py.File(hdf5_path, "a") as f:
        for i, system_name in enumerate(systems):
            if system_name not in ipsae_df.index:
                n_missing += 1
                missing_log.append(system_name)
                continue

            row = ipsae_df.loc[system_name]
            for col in METRIC_COLS:
                if col in ipsae_df.columns:
                    val = row[col]
                    f[col][i] = float(val) if pd.notna(val) else np.nan

            n_matched += 1

    return {
        "matched": n_matched,
        "missing": n_missing,
        "missing_list": missing_log
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    hdf5_path = Path(args.hdf5_path)
    csv_path  = Path(args.ipsae_csv)

    # ----------------------------------------------------------------
    # Validate
    # ----------------------------------------------------------------
    if not hdf5_path.exists():
        log.error(f"HDF5 not found: {hdf5_path}")
        sys.exit(1)

    if not csv_path.exists():
        log.error(f"ipSAE CSV not found: {csv_path}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    ipsae_df = load_ipsae_csv(csv_path)

    # Check all expected metric columns are present
    missing_cols = [c for c in METRIC_COLS if c not in ipsae_df.columns]
    if missing_cols:
        log.warning(f"  Missing metric columns in CSV: {missing_cols}")
        log.warning(f"  These will be skipped")

    systems = get_hdf5_systems(hdf5_path)
    log.info(f"HDF5 contains {len(systems)} systems")

    # ----------------------------------------------------------------
    # Initialise metric datasets if needed
    # ----------------------------------------------------------------
    init_metric_datasets(hdf5_path, len(systems))

    # ----------------------------------------------------------------
    # Write metrics
    # ----------------------------------------------------------------
    log.info("Writing metrics to HDF5...")
    counts = write_metrics(hdf5_path, systems, ipsae_df)

    # ----------------------------------------------------------------
    # Log missing systems
    # ----------------------------------------------------------------
    if counts["missing_list"]:
        missing_path = hdf5_path.parent / "missing_metrics.txt"
        with open(missing_path, "w") as f:
            f.write("\n".join(counts["missing_list"]))
        log.warning(
            f"  {counts['missing']} systems not found in CSV — "
            f"metrics left as NaN\n"
            f"  Full list: {missing_path}"
        )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("COMPLETE")
    log.info(f"  Total systems in HDF5 : {len(systems)}")
    log.info(f"  Metrics written       : {counts['matched']}")
    log.info(f"  Missing from CSV      : {counts['missing']}")
    log.info(f"  HDF5                  : {hdf5_path}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add ipSAE metrics to embeddings HDF5"
    )
    parser.add_argument(
        "--hdf5-path",
        default="embeddings.h5",
        help="Path to embeddings HDF5 (default: embeddings.h5)"
    )
    parser.add_argument(
        "--ipsae-csv",
        required=True,
        help="Path to ipSAE summary CSV"
    )

    args = parser.parse_args()
    main(args)