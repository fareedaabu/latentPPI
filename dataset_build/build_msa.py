#!/usr/bin/env python3
"""
build_msa_cache_server.py

Generate per-sequence MSAs for all UNIQUE proteins appearing in a PPI pair set,
using the ColabFold remote MSA server via colabfold_batch --msa-only.

WHY PER-SEQUENCE (not per-pair):
  The same protein recurs across many pairs in a proteome-scale set. Searching
  each unique accession ONCE and caching the result avoids redundant server
  queries. Cache key = accession verbatim (isoform suffix preserved, e.g.
  'A1KXE4-2'), because an accession without its isoform suffix is NOT unique
  to a sequence.

WHY colabfold_batch --msa-only (not local colabfold_search):
  Local colabfold_search requires ColabFold-format MMseqs2 databases (UniRef30
  + colabfold_envdb in profile DB format) which are not available. The AF3-
  native databases are not compatible with colabfold_search. The remote server
  is the correct path for ~11k unique sequences (well within fair-use limits)
  and is far faster than AF3's Jackhmmer pipeline for dataset-building.

OUTPUT PER SEQUENCE:
  colabfold_batch --msa-only --af3-json produces:
    <accession>.json  — AF3-compatible single-chain input with MSA embedded
                        as a string (--af3-msa-as-path is not in this build)
    <accession>.a3m   — the raw MSA file (written regardless of --af3-json)
  Both are filed into the cache. The a3m is the completion marker (more
  reliable than the JSON since it is written by the search engine itself).

STORAGE NOTE:
  Embedded MSA strings make cache JSONs 1-5 MB each (~11-55 GB for 11k
  proteins). Pair JSONs built from this cache should be treated as ephemeral
  and deleted after successful AF3 inference — they are not needed once
  metrics and embeddings are extracted.

WORKFLOW:
  1. Read pair records (JSON list with uniprot_a/seq_a/uniprot_b/seq_b/label).
  2. Build unique (accession -> sequence), asserting one sequence per accession.
  3. Skip accessions already cached (idempotent / resumable).
  4. Write a temp FASTA of only the not-yet-cached sequences.
     Headers are verbatim accessions — NO _A/_B suffix (chain identity is a
     pair-construction concept, not a per-protein MSA concept).
  5. Run colabfold_batch --msa-only --af3-json.
  6. File each JSON + a3m into the cache keyed by accession.
  7. Write a coverage report before you build any pair JSONs.

USAGE:
  python build_msa.py --pairs positives.json negatives.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ======================================================================
#      define the cahce and colabfold command
# ======================================================================
CACHE_DIR = "ppi_dataset/inputs"   # where per-accession JSONs + a3ms are stored
COLABFOLD_BATCH = "colabfold_batch"       # on PATH in the colabfold env; override if needed
# ======================================================================

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def log(msg: str) -> None:
    print(f"[build_msa_cache] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("RUN: " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def load_pairs(paths: list[Path]) -> list[dict]:
    """Load pair records from one or more JSON files (each a list of dicts)."""
    records = []
    for path in paths:
        if not path.is_file():
            log(f"ERROR: pairs file not found: {path}")
            sys.exit(2)
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            log(f"ERROR: expected a JSON list in {path}, got {type(data).__name__}")
            sys.exit(2)
        log(f"loaded {len(data)} records from {path.name}")
        records.extend(data)
    return records


def collect_unique_sequences(records: list[dict]) -> dict[str, str]:
    """Build accession -> sequence, asserting one sequence per accession.

    Exits if the same accession maps to two different sequences — that
    silently corrupts the chain-length contract downstream.
    """
    acc_to_seq: dict[str, str] = {}
    conflicts: list[str] = []

    def add(acc: str, seq: str) -> None:
        acc = (acc or "").strip()
        seq = (seq or "").strip().upper()
        if not acc or not seq:
            return
        if acc in acc_to_seq:
            if acc_to_seq[acc] != seq:
                conflicts.append(acc)
        else:
            acc_to_seq[acc] = seq

    for r in records:
        add(r.get("uniprot_a", ""), r.get("seq_a", ""))
        add(r.get("uniprot_b", ""), r.get("seq_b", ""))

    if conflicts:
        uniq = sorted(set(conflicts))
        log(f"ERROR: {len(uniq)} accession(s) map to MORE THAN ONE sequence.")
        log("       This breaks per-accession caching. Fix the input data.")
        for acc in uniq[:20]:
            log(f"         conflicting accession: {acc}")
        if len(uniq) > 20:
            log(f"         ... and {len(uniq) - 20} more")
        sys.exit(2)

    return acc_to_seq


def flag_non_standard(acc_to_seq: dict[str, str]) -> dict[str, str]:
    """Return {accession: offending_residues} for non-standard AA sequences."""
    return {
        acc: "".join(sorted(set(seq) - STANDARD_AA))
        for acc, seq in acc_to_seq.items()
        if set(seq) - STANDARD_AA
    }


def safe_cache_name(acc: str) -> str:
    """Filesystem-safe stem: isoform dash preserved (A1KXE4-2 stays A1KXE4-2)."""
    return acc.replace("/", "_").replace(":", "_").replace(".", "_")


def is_cached(acc: str, cache_dir: Path) -> bool:
    """Cache is complete when the a3m exists (written by the search engine).
    The JSON may also exist, but the a3m is the more reliable completion marker.
    """
    stem = safe_cache_name(acc)
    return any(cache_dir.glob(f"{stem}*.a3m"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build per-sequence MSA cache via ColabFold remote server."
    )
    p.add_argument(
        "--pairs", type=Path, nargs="+", required=True,
        help="One or more JSON files of pair records "
             "(uniprot_a/seq_a/uniprot_b/seq_b/label)."
    )
    p.add_argument(
        "--cache-dir", type=Path, default=Path(CACHE_DIR),
        help="Per-accession MSA cache directory."
    )
    p.add_argument(
        "--colabfold-batch", default=COLABFOLD_BATCH,
        help="colabfold_batch executable (default: 'colabfold_batch' on PATH)."
    )
    p.add_argument(
        "--skip-non-standard", action="store_true",
        help="Exclude sequences with non-standard residues from the search."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be searched without running colabfold_batch."
    )
    args = p.parse_args()

    if "PLACEHOLDER" in str(args.cache_dir):
        log("ERROR: PLACEHOLDER cache dir is still set. Edit the script header "
            "or pass --cache-dir on the command line.")
        sys.exit(2)

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1-2. Load and dedup ----
    records = load_pairs(args.pairs)
    acc_to_seq = collect_unique_sequences(records)
    log(f"unique accessions across all pairs: {len(acc_to_seq)}")

    # ---- non-standard residue handling ----
    flagged = flag_non_standard(acc_to_seq)
    if flagged:
        log(f"{len(flagged)} accession(s) contain non-standard residues.")
        nonstd_log = args.cache_dir / "non_standard_accessions.tsv"
        with open(nonstd_log, "w") as f:
            f.write("accession\tnon_standard_residues\n")
            for acc, bad in sorted(flagged.items()):
                f.write(f"{acc}\t{bad}\n")
        log(f"  logged to {nonstd_log}")
        if args.skip_non_standard:
            for acc in flagged:
                acc_to_seq.pop(acc, None)
            log(f"  excluded {len(flagged)} (--skip-non-standard); "
                f"remaining: {len(acc_to_seq)}")
        else:
            log("  NOT excluded (pass --skip-non-standard to drop them). "
                "AF3 may reject these at inference time.")

    # ---- 3. Determine what still needs searching ----
    todo = {
        acc: seq for acc, seq in acc_to_seq.items()
        if not is_cached(acc, args.cache_dir)
    }
    already = len(acc_to_seq) - len(todo)
    log(f"already cached: {already} | to search: {len(todo)}")

    if not todo:
        log("Nothing to search; cache is complete.")
        write_coverage(acc_to_seq, args.cache_dir)
        return

    if args.dry_run:
        log(f"DRY RUN: would submit {len(todo)} sequence(s) to the "
            "ColabFold server (not running colabfold_batch).")
        write_coverage(acc_to_seq, args.cache_dir, dry_run=True)
        return

    # ---- 4. Temp FASTA of not-yet-cached sequences ----
    # Headers are verbatim accessions — NO _A/_B suffix.
    # Chain identity is assigned in build_af3_inputs.py, not here.
    work = Path(tempfile.mkdtemp(prefix="msa_server_"))
    query_fasta = work / "to_search.fasta"
    search_out = work / "search_out"
    search_out.mkdir(parents=True, exist_ok=True)

    with open(query_fasta, "w") as f:
        for acc, seq in todo.items():
            f.write(f">{acc}\n{seq}\n")
    log(f"wrote query FASTA with {len(todo)} sequences: {query_fasta}")

    # ---- 5. colabfold_batch: remote server, MSA only, AF3 JSON ----
    # --msa-only : skip structure prediction entirely
    # --af3-json : emit per-sequence AF3-compatible input JSON
    #              MSA is embedded as a string (--af3-msa-as-path not in
    #              this build, but the a3m is written alongside regardless)
    cmd = [
        args.colabfold_batch,
        "--msa-only",
        "--af3-json",
        str(query_fasta),
        str(search_out),
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: colabfold_batch failed (exit {e.returncode}). "
            f"Intermediate output preserved at: {search_out}")
        log("The cache is resumable: re-run after fixing the issue and "
            "only not-yet-cached sequences will be resubmitted.")
        sys.exit(1)

    # ---- 6. File results into the cache ----
    # colabfold_batch --af3-json writes one JSON + one a3m per query sequence.
    # We recover the accession from the JSON's 'name' field (most reliable),
    # fall back to filename stem if absent, and attempt to recover from
    # ColabFold name-normalization (lowercase, separator changes) if needed.
    produced_jsons = list(search_out.rglob("*.json"))
    produced_a3ms = list(search_out.rglob("*.a3m"))
    log(f"colabfold_batch produced {len(produced_jsons)} JSON(s) "
        f"and {len(produced_a3ms)} a3m(s) in {search_out}")

    cached_now = 0
    for jf in produced_jsons:
        # Recover accession
        acc = None
        try:
            with open(jf) as f:
                j = json.load(f)
            acc = (j.get("name") or "").strip() or None
        except Exception:
            pass
        if acc is None:
            acc = jf.stem

        # Normalization recovery: ColabFold may lowercase or alter separators
        if acc not in todo:
            match = next(
                (a for a in todo
                 if safe_cache_name(a).lower() == safe_cache_name(acc).lower()),
                None
            )
            if match:
                log(f"  INFO: normalized '{acc}' -> '{match}' "
                    "(case/separator mismatch)")
                acc = match
            else:
                log(f"  WARNING: produced JSON '{jf.name}' did not map to "
                    f"any queried accession (parsed: '{acc}'). "
                    "Filing under parsed name.")

        stem = safe_cache_name(acc)

        # Copy JSON
        dest_json = args.cache_dir / f"{stem}.json"
        shutil.copy2(jf, dest_json)
        log(f"  cached JSON : {dest_json.name}")

        # Copy sibling a3m (same stem, different suffix)
        sibling_a3m = jf.with_suffix(".a3m")
        if sibling_a3m.exists():
            dest_a3m = args.cache_dir / f"{stem}.a3m"
            shutil.copy2(sibling_a3m, dest_a3m)
            log(f"  cached a3m  : {dest_a3m.name}")
        else:
            # Search for any a3m with matching stem (name normalization)
            matches = list(jf.parent.glob(f"{jf.stem}*.a3m"))
            for m in matches:
                dest_a3m = args.cache_dir / f"{stem}.a3m"
                shutil.copy2(m, dest_a3m)
                log(f"  cached a3m  : {dest_a3m.name} (from {m.name})")

        cached_now += 1

    log(f"cached {cached_now} new accession result(s) into {args.cache_dir}")

    # ---- 7. Coverage report ----
    write_coverage(acc_to_seq, args.cache_dir)

    # Clean temp workspace only on success
    shutil.rmtree(work, ignore_errors=True)
    log("done.")


def write_coverage(
    acc_to_seq: dict[str, str],
    cache_dir: Path,
    dry_run: bool = False
) -> None:
    """Write TSV of which accessions have a cached a3m and which don't."""
    report = cache_dir / ("coverage_dryrun.tsv" if dry_run else "coverage.tsv")
    have, missing = [], []
    for acc in sorted(acc_to_seq):
        (have if is_cached(acc, cache_dir) else missing).append(acc)
    with open(report, "w") as f:
        f.write("accession\tcached\tlength\n")
        for acc in sorted(acc_to_seq):
            f.write(f"{acc}\t{int(is_cached(acc, cache_dir))}"
                    f"\t{len(acc_to_seq[acc])}\n")
    log(f"coverage: {len(have)} cached / {len(missing)} missing -> {report}")
    if missing:
        log(f"  WARNING: {len(missing)} accession(s) still lack an a3m.")
        log("  With AF3 --norun_data_pipeline, pairs using these run "
            "single-sequence. Resolve before building pair JSONs.")


if __name__ == "__main__":
    main()