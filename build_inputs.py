#!/usr/bin/env python3
"""
Build AF3 input JSONs from a PPI pair summary TSV.
=====================================================================
Reads a TSV with uniprot_a/uniprot_b columns, finds matching .a3m
MSA files for each chain, and writes AF3 input JSONs directly.

System names: {uniprot_a}_{uniprot_b}

AF3 JSON format:
  {
    "name": "P04637_Q9Y4K3",
    "dialect": "alphafold3",
    "version": 1,
    "modelSeeds": [1],
    "sequences": [
      {"protein": {"id": "A", "sequence": "...",
                   "unpairedMsaPath": "../msa_cache/P04637.a3m",
                   "pairedMsa": " ", "templates": []}},
      {"protein": {"id": "B", "sequence": "...",
                   "unpairedMsaPath": "../msa_cache/Q9Y4K3.a3m",
                   "pairedMsa": " ", "templates": []}}
    ]
  }

Usage:
    # Positives
    python build_inputs.py \\
        --tsv ./dataset/dataset_summary.tsv \\
        --msa-dir ./msa_cache \\
        --output-dir ./af3_inputs/positives

"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSA helpers
# ---------------------------------------------------------------------------

def find_msa(uniprot_id: str, msa_dir: Path) -> Path | None:
    """
    Locate {uniprot_id}.a3m in msa_dir.
    Returns Path if found, None otherwise.
    """
    candidate = msa_dir / f"{uniprot_id}.a3m"
    return candidate if candidate.exists() else None


def load_sequence_cache(cache_path: Path) -> dict[str, str]:
    """Load accession -> sequence cache JSON, returning {} if unavailable."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.warning(f"sequence cache is not a dict: {cache_path}")
            return {}
        return {str(k).strip(): str(v).strip().upper() for k, v in data.items() if str(v).strip()}
    except Exception as e:
        log.warning(f"failed to load sequence cache {cache_path}: {e}")
        return {}


def extract_first_sequence(a3m_path: Path) -> str:
    """
    Extract the query sequence from the first entry of an .a3m file.
    Strips lowercase insertion columns and gap characters.
    Returns sequence string or empty string on failure.
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
                    break   # hit second header — done
                in_first = True
                continue
            if in_first:
                seq_lines.append(line)

    seq = "".join(seq_lines)
    # Remove lowercase insertion columns and gaps
    seq = "".join(c for c in seq if c.isupper() or c == "-")
    seq = seq.replace("-", "")
    return seq


def build_msa_sequence_index(msa_dir: Path) -> tuple[dict[str, Path], dict[str, Path], int]:
    """
    Build lookup maps for available MSAs:
      - accession_to_msa: accession stem -> MSA path
      - seq_to_msa: query sequence -> representative MSA path
    Returns both maps plus a count of MSAs that could not be parsed.
    """
    accession_to_msa: dict[str, Path] = {}
    seq_to_msa: dict[str, Path] = {}
    parse_failures = 0

    # Deterministic ordering makes sequence-to-MSA assignment stable.
    for msa_path in sorted(msa_dir.glob("*.a3m")):
        accession_to_msa[msa_path.stem] = msa_path
        seq = extract_first_sequence(msa_path)
        if not seq:
            parse_failures += 1
            continue
        if seq not in seq_to_msa:
            seq_to_msa[seq] = msa_path

    return accession_to_msa, seq_to_msa, parse_failures


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------

def write_af3_json(uniprot_a: str,
                   uniprot_b: str,
                   seq_a: str,
                   seq_b: str,
                   msa_path_a: Path,
                   msa_path_b: Path,
                   output_path: Path,
                   model_seed: int = 1,
                   absolute_msa_paths: bool = False) -> None:
    """
    Write a single AF3 input JSON for a heterodimeric pair.
    Chain A = uniprot_a, Chain B = uniprot_b.
    Dataset inputs use the sibling msa_cache by default. Screen inputs can
    reference the original MSA files with absolute_msa_paths=True.
    """
    if absolute_msa_paths:
        msa_ref_a = str(msa_path_a.resolve())
        msa_ref_b = str(msa_path_b.resolve())
    else:
        msa_ref_a = f"../msa_cache/{msa_path_a.name}"
        msa_ref_b = f"../msa_cache/{msa_path_b.name}"
    data = {
        "name":       f"{uniprot_a}_{uniprot_b}",
        "dialect":    "alphafold3",
        "version":    1,
        "modelSeeds": [model_seed],
        "sequences": [
            {
                "protein": {
                    "id":              "A",
                    "sequence":        seq_a,
                    "unpairedMsaPath": msa_ref_a,
                    "pairedMsa":       " ",
                    "templates":       []
                }
            },
            {
                "protein": {
                    "id":              "B",
                    "sequence":        seq_b,
                    "unpairedMsaPath": msa_ref_b,
                    "pairedMsa":       " ",
                    "templates":       []
                }
            }
        ]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    tsv_path   = Path(args.tsv)
    msa_dir    = Path(args.msa_dir)
    output_dir = Path(args.output_dir)

    # ----------------------------------------------------------------
    # Validate inputs
    # ----------------------------------------------------------------
    if not tsv_path.exists():
        log.error(f"TSV not found: {tsv_path}")
        sys.exit(1)

    if not msa_dir.exists():
        log.error(f"MSA directory not found: {msa_dir}")
        sys.exit(1)

    if args.sequence_cache:
        seq_cache_path = Path(args.sequence_cache)
    else:
        # Default to sibling cache produced during curation.
        seq_cache_path = tsv_path.parent / "sequence_cache.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    msa_files = list(msa_dir.glob("*.a3m"))
    accession_to_msa, seq_to_msa, parse_failures = build_msa_sequence_index(msa_dir)
    seq_cache = load_sequence_cache(seq_cache_path)
    log.info("=" * 60)
    log.info(f"TSV            : {tsv_path}")
    log.info(f"MSA directory  : {msa_dir} ({len(msa_files)} .a3m files)")
    log.info(f"Sequence cache : {seq_cache_path} ({len(seq_cache)} entries)")
    log.info(f"MSA seq index  : {len(seq_to_msa)} unique query sequences")
    if parse_failures:
        log.warning(f"{parse_failures} .a3m file(s) had no parseable query sequence")
    log.info(f"Output dir     : {output_dir}")
    log.info("=" * 60)

    # ----------------------------------------------------------------
    # Load TSV
    # ----------------------------------------------------------------
    df = pd.read_csv(tsv_path, sep="\t")
    log.info(f"Loaded {len(df)} pairs from {tsv_path.name}")

    if "uniprot_a" not in df.columns or "uniprot_b" not in df.columns:
        log.error(
            f"TSV missing uniprot_a/uniprot_b columns. "
            f"Found: {list(df.columns)}"
        )
        sys.exit(1)

    has_curated_msa_cols = {"msa_acc_a", "msa_acc_b"}.issubset(df.columns)
    if has_curated_msa_cols:
        log.info("Found curated MSA mapping columns (msa_acc_a/msa_acc_b); these will be preferred.")
    else:
        log.info("No curated MSA mapping columns found; using direct accession lookup + sequence fallback.")

    # ----------------------------------------------------------------
    # Build JSONs
    # ----------------------------------------------------------------
    n_success     = 0
    n_skipped     = 0
    n_missing_msa = 0
    n_failed      = 0
    missing_log   = []
    assigned_map: dict[str, tuple[str, str]] = {}

    def _clean_optional(value) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text

    def resolve_msa_with_fallback(uniprot_id: str,
                                  preferred_msa_acc: str = "",
                                  preferred_source: str = "") -> Path | None:
        """
        Resolve an accession's MSA.
        Priority: curated summary mapping -> direct accession -> sequence fallback.
        """
        preferred_msa_acc = preferred_msa_acc.strip()
        if preferred_msa_acc:
            curated = accession_to_msa.get(preferred_msa_acc)
            if curated:
                if preferred_msa_acc != uniprot_id:
                    src = preferred_source.strip() or "summary_mapped"
                    assigned_map.setdefault(uniprot_id, (preferred_msa_acc, src))
                return curated

        direct = accession_to_msa.get(uniprot_id)
        if direct:
            return direct

        seq = seq_cache.get(uniprot_id, "")
        if not seq:
            return None

        mapped = seq_to_msa.get(seq)
        if not mapped:
            return None

        assigned_map.setdefault(uniprot_id, (mapped.stem, "sequence_fallback"))
        return mapped

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building JSONs"):
        uniprot_a   = str(row["uniprot_a"]).strip()
        uniprot_b   = str(row["uniprot_b"]).strip()
        system_name = f"{uniprot_a}_{uniprot_b}"
        json_path   = output_dir / f"{system_name}.json"

        preferred_msa_a = _clean_optional(row.get("msa_acc_a", "")) if has_curated_msa_cols else ""
        preferred_msa_b = _clean_optional(row.get("msa_acc_b", "")) if has_curated_msa_cols else ""
        preferred_src_a = _clean_optional(row.get("msa_source_a", "")) if "msa_source_a" in df.columns else ""
        preferred_src_b = _clean_optional(row.get("msa_source_b", "")) if "msa_source_b" in df.columns else ""

        # Skip if already built and not refreshing
        if json_path.exists() and not args.refresh:
            n_skipped += 1
            log.debug(f"  {system_name}: already exists, skipping")
            continue

        # Locate MSA files
        msa_path_a = resolve_msa_with_fallback(uniprot_a, preferred_msa_a, preferred_src_a)
        msa_path_b = resolve_msa_with_fallback(uniprot_b, preferred_msa_b, preferred_src_b)

        if not msa_path_a or not msa_path_b:
            missing = []
            if not msa_path_a:
                missing.append(f"{uniprot_a}.a3m")
            if not msa_path_b:
                missing.append(f"{uniprot_b}.a3m")
            missing_log.append(f"{system_name}: missing {', '.join(missing)}")
            n_missing_msa += 1
            continue

        # Extract query sequences from MSA first entries
        seq_a = extract_first_sequence(msa_path_a)
        seq_b = extract_first_sequence(msa_path_b)

        if not seq_a or not seq_b:
            log.warning(f"  {system_name}: could not extract sequence from MSA")
            n_failed += 1
            continue

        # Write AF3 JSON
        try:
            write_af3_json(
                uniprot_a=uniprot_a,
                uniprot_b=uniprot_b,
                seq_a=seq_a,
                seq_b=seq_b,
                msa_path_a=msa_path_a,
                msa_path_b=msa_path_b,
                output_path=json_path
            )
            n_success += 1
        except Exception as e:
            log.warning(f"  {system_name}: failed to write JSON — {e}")
            n_failed += 1

    # ----------------------------------------------------------------
    # Write missing MSA log
    # ----------------------------------------------------------------
    if missing_log:
        missing_path = output_dir / "missing_msas.txt"
        with open(missing_path, "w") as f:
            f.write("\n".join(missing_log))
        log.warning(
            f"{n_missing_msa} pairs skipped — missing MSAs\n"
            f"  Full list: {missing_path}"
        )

    assigned_path = output_dir / "assigned_msas.txt"
    with open(assigned_path, "w") as f:
        f.write("accession\tmapped_to_msa\tsource\n")
        for accession, (mapped_to, source) in sorted(assigned_map.items()):
            f.write(f"{accession}\t{mapped_to}.a3m\t{source}\n")
    log.info(f"Assigned MSA mappings: {len(assigned_map)} accessions -> {assigned_path}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("COMPLETE")
    log.info(f"  Total pairs   : {len(df)}")
    log.info(f"  JSONs built   : {n_success}")
    log.info(f"  Already exist : {n_skipped}")
    log.info(f"  Missing MSAs  : {n_missing_msa}")
    log.info(f"  MSA assigned  : {len(assigned_map)} accessions")
    log.info(f"  Failures      : {n_failed}")
    log.info(f"  Output        : {output_dir}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build AF3 input JSONs from a PPI pair summary TSV"
    )
    parser.add_argument(
        "--tsv",
        required=True,
        help="Path to summary TSV with uniprot_a/uniprot_b columns"
    )
    parser.add_argument(
        "--msa-dir",
        required=True,
        help="Directory containing per-protein .a3m files ({uniprot_id}.a3m)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated AF3 JSON files"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate JSONs even if they already exist"
    )
    parser.add_argument(
        "--sequence-cache",
        default="",
        help="Optional accession->sequence JSON cache for sequence-based MSA fallback. "
             "Default: <dirname of --tsv>/sequence_cache.json"
    )

    args = parser.parse_args()
    main(args)
