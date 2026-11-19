#!/usr/bin/env python3
"""
build_mmseqs_db.py

One-time build of an MMseqs2 sequence database from a UniRef90 FASTA, for use
with colabfold_search in local (no-server) MSA generation.

DESIGN CHOICES (per pipeline decisions):
  - createdb ONLY, no createindex. The precomputed k-mer index would add
    ~200-300 GB and only pays off for repeated/interactive querying held in
    RAM. This is a one-time bulk cache build, so we skip the index: smaller
    footprint, no big-RAM node needed. Search is slower per query but still
    far faster than Jackhmmer/server, and the cost is paid once.
  - Idempotent: if the DB already exists, the build is skipped.
  - Reports the real on-disk size after building so you can confirm before
    proceeding to the search step.

USAGE:
  python build_mmseqs_db.py
  (edit the PLACEHOLDER paths below first, or override via CLI flags)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ======================================================================
# PLACEHOLDER PATHS — fill these in (or override on the command line)
# ======================================================================
MMSEQS_BIN = "mmseqs"                       # e.g. /path/to/mmseqs  (or just "mmseqs" if on PATH)
UNIREF90_FASTA = "/path/to/uniref90_2022_05.fa"
MMSEQS_DB_DIR = "/path/to/mmseqs_db"             # directory to hold the built DB (~60-80 GB)
DB_NAME = "uniref90_db"                                 # basename for the MMseqs2 DB files
# ======================================================================


def log(msg: str) -> None:
    print(f"[build_mmseqs_db] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("RUN: " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def dir_size_human(path: Path) -> str:
    """Return human-readable size of a directory's contents via du."""
    try:
        out = subprocess.run(
            ["du", "-sh", str(path)], capture_output=True, text=True, check=True
        )
        return out.stdout.split()[0]
    except Exception as e:
        return f"<du failed: {e}>"


def db_exists(db_path: Path) -> bool:
    """An MMseqs2 createdb leaves <db> and <db>.dbtype among other files.
    We treat the presence of <db>.dbtype as the completion marker."""
    return db_path.with_suffix(".dbtype").exists() or Path(str(db_path) + ".dbtype").exists()


def main() -> None:
    p = argparse.ArgumentParser(description="One-time MMseqs2 createdb build (no index).")
    p.add_argument("--mmseqs", default=MMSEQS_BIN, help="Path to mmseqs binary (or 'mmseqs' if on PATH).")
    p.add_argument("--fasta", type=Path, default=Path(UNIREF90_FASTA), help="UniRef90 FASTA to build from.")
    p.add_argument("--db-dir", type=Path, default=Path(MMSEQS_DB_DIR), help="Directory to hold the built DB.")
    p.add_argument("--db-name", default=DB_NAME, help="Basename for the MMseqs2 DB.")
    p.add_argument("--force", action="store_true", help="Rebuild even if the DB already exists.")
    args = p.parse_args()

    # ---- Validate inputs (fail loud and early) ----
    for placeholder in (args.mmseqs, str(args.fasta), str(args.db_dir)):
        if "PLACEHOLDER" in placeholder:
            log("ERROR: a PLACEHOLDER path is still set. Edit the script header "
                "or pass --mmseqs / --fasta / --db-dir on the command line.")
            sys.exit(2)

    if not args.fasta.is_file():
        log(f"ERROR: FASTA not found: {args.fasta}")
        sys.exit(2)

    args.db_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.db_dir / args.db_name

    log(f"mmseqs binary : {args.mmseqs}")
    log(f"source FASTA  : {args.fasta}  ({dir_size_human(args.fasta) if args.fasta.is_dir() else _file_size_human(args.fasta)})")
    log(f"DB target     : {db_path}")

    # ---- Idempotency ----
    if db_exists(db_path) and not args.force:
        log("DB already exists (found .dbtype marker). Skipping build. Use --force to rebuild.")
        log(f"Current DB dir size: {dir_size_human(args.db_dir)}")
        return

    if args.force and db_exists(db_path):
        log("--force set: removing existing DB files before rebuild.")
        for f in args.db_dir.glob(f"{args.db_name}*"):
            log(f"  removing {f.name}")
            f.unlink()

    # ---- createdb (NO createindex, by design) ----
    log("Starting `mmseqs createdb` (no index will be built).")
    run([args.mmseqs, "createdb", str(args.fasta), str(db_path)])

    log("createdb complete.")
    log(f"On-disk DB size: {dir_size_human(args.db_dir)}")
    log("NOTE: No k-mer index was built (by design). colabfold_search will read "
        "the DB from disk; --db-load-mode is a no-op without an index.")
    log("If searches feel I/O-bound, confirm this volume is SSD, not spinning disk.")


def _file_size_human(path: Path) -> str:
    try:
        out = subprocess.run(["du", "-h", str(path)], capture_output=True, text=True, check=True)
        return out.stdout.split()[0]
    except Exception as e:
        return f"<du failed: {e}>"


if __name__ == "__main__":
    main()