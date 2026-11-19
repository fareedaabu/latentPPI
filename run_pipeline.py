#!/usr/bin/env python3
"""Prepare AF3 pair inputs and commands ordered by total residue count.

This is an input-preparation pipeline. It does not launch GPU inference.
Screen JSONs reference the query MSA in the screen folder and proteome MSAs
in their original directory; no MSA copies are made.
af3_lengths.csv records the chain lengths in command-file order.
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import shlex
from pathlib import Path

from build_inputs import extract_first_sequence, write_af3_json


LOG = logging.getLogger("run_pipeline")
MANIFEST_FIELDS = ("system_name", "af3_json", "af3_out_dir")
STANDARD_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")


def existing_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{description} not found: {path}")
    return path.resolve()


def query_sequence(path: Path) -> str:
    sequence = extract_first_sequence(path)
    if not sequence:
        raise ValueError(f"No query sequence found in {path}")
    return sequence


def create_query_msa(fasta: Path, sequence: str, root: Path, force: bool) -> Path:
    """Create a query-only A3M in the screen folder when no query MSA was supplied."""
    digest = hashlib.sha256(str(fasta.resolve()).encode()).hexdigest()[:12]
    target = root / f"{fasta.stem}_{digest}_query.a3m"
    if force or not target.exists():
        target.write_text(f">{fasta.stem}\n{sequence}\n")
    return target


def load_manifest(path: Path, recreate: bool) -> dict[str, dict[str, str]]:
    if recreate or not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"system_name", "af3_json", "af3_out_dir"}.issubset(reader.fieldnames):
            raise ValueError(f"Not an AF3 jobs manifest: {path}")
        return {row["system_name"]: row for row in reader if row.get("system_name")}


def save_manifest(path: Path, rows: dict[str, dict[str, str]], recreate: bool) -> None:
    fields = MANIFEST_FIELDS
    if path.exists() and not recreate:
        with path.open(newline="") as handle:
            old_fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
        fields = tuple(dict.fromkeys((*MANIFEST_FIELDS, *old_fields)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows.values():
            writer.writerow({field: row.get(field, "") for field in fields})


def command_for(json_path: Path, output_dir: Path, recycles: int, samples: int) -> str:
    parts = [
        "run_alphafold.py", "--norun_data_pipeline",
        f"--json_path={json_path}", f"--output_dir={output_dir}",
        f"--num_recycles={recycles}", f"--num_diffusion_samples={samples}",
        "--force_output_dir",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def command_json_path(command: str, base_dir: Path) -> Path | None:
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token.startswith("--json_path="):
            value = token.split("=", 1)[1]
        elif token == "--json_path" and index + 1 < len(tokens):
            value = tokens[index + 1]
        else:
            continue
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base_dir / path).resolve()
    return None


def af3_lengths(json_path: Path) -> tuple[str, int, int]:
    with json_path.open() as handle:
        data = json.load(handle)
    proteins = [entry["protein"] for entry in data["sequences"] if "protein" in entry]
    if len(proteins) != 2:
        raise ValueError(f"Expected two protein chains in {json_path}")
    return data["name"], len(proteins[0]["sequence"]), len(proteins[1]["sequence"])


def save_commands(path: Path, rows: dict[str, dict[str, str]], args: argparse.Namespace) -> int:
    commands = [] if args.recreate_commands or not path.exists() else [
        line for line in path.read_text().splitlines() if line.strip()
    ]
    known = set(commands)
    for row in rows.values():
        json_path = Path(row.get("af3_json", ""))
        out_dir = Path(row.get("af3_out_dir", ""))
        if not json_path.is_absolute():
            json_path = path.parent / json_path
        if not out_dir.is_absolute():
            out_dir = path.parent / out_dir
        if json_path.is_file() and str(out_dir) != ".":
            command = command_for(json_path, out_dir.parent, args.af3_num_recycles,
                                  args.af3_num_diffusion_samples)
            if command not in known:
                commands.append(command)
                known.add(command)

    lengths_cache: dict[Path, tuple[str, int, int]] = {}
    measured = []
    for command in commands:
        json_path = command_json_path(command, path.parent)
        if json_path and json_path.is_file():
            if json_path not in lengths_cache:
                lengths_cache[json_path] = af3_lengths(json_path)
            system_name, len_a, len_b = lengths_cache[json_path]
            if system_name in getattr(args, "excluded_systems", set()):
                continue
            measured.append((len_a + len_b, system_name, len_a, len_b, json_path, command))
        else:
            LOG.warning("Cannot determine length; placing command last: %s", command)
            measured.append((None, json_path.stem if json_path else "", None, None,
                             json_path, command))

    measured.sort(key=lambda item: (item[0] is None,
                                    item[0] if item[0] is not None else 0,
                                    item[1].casefold(), item[5]))
    path.write_text("".join(f"{item[5]}\n" for item in measured))
    lengths_csv = path.parent / "af3_lengths.csv"
    with lengths_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("line_number", "system_name", "chain_a_length", "chain_b_length",
                         "total_length", "json_path"))
        for line_number, (total, name, len_a, len_b, json_path, _) in enumerate(measured, 1):
            writer.writerow((line_number, name, len_a if len_a is not None else "",
                             len_b if len_b is not None else "",
                             total if total is not None else "", json_path or ""))
    return len(measured)


def prepare_pair(system_name: str, fasta: Path, msa_a: Path | None, msa_b: Path,
                 root: Path, args: argparse.Namespace, candidate=None) -> dict[str, str] | None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", system_name) or system_name in {".", ".."}:
        raise ValueError(f"Invalid system name: {system_name}")

    seq_a = query_sequence(msa_a or fasta)
    seq_b = query_sequence(msa_b)
    if candidate is not None:
        candidate.update(chain_a_length=len(seq_a), chain_b_length=len(seq_b),
                         total_length=len(seq_a) + len(seq_b), homodimer=seq_a == seq_b,
                         homodimer_rule_applied=False, preprocessing_status="ready")
    if not args.no_filter_non_standard and (
        set(seq_a) - STANDARD_RESIDUES or set(seq_b) - STANDARD_RESIDUES
    ):
        LOG.info("Skipping %s: nonstandard residues in query sequence", system_name)
        if candidate is not None:
            candidate["preprocessing_status"] = "skipped_nonstandard"
        return None
    if args.max_identity < 100:
        identity = 100 * sum(a == b for a, b in zip(seq_a, seq_b)) / min(len(seq_a), len(seq_b))
        if identity > args.max_identity:
            LOG.info("Skipping %s: sequence identity %.1f%% exceeds %.1f%%", system_name,
                     identity, args.max_identity)
            if candidate is not None:
                candidate["preprocessing_status"] = "skipped_identity"
            return None

    json_dir = root / "af3_inputs"
    json_dir.mkdir(parents=True, exist_ok=True)
    query_msa = msa_a or create_query_msa(fasta, seq_a, root, args.force)
    json_path = json_dir / f"{system_name}.json"
    expected_msa_paths = [str(query_msa.resolve()), str(msa_b.resolve())]
    if json_path.exists() and not args.force:
        with json_path.open() as handle:
            existing_json = json.load(handle)
        current_msa_paths = [entry.get("protein", {}).get("unpairedMsaPath")
                             for entry in existing_json.get("sequences", [])]
    else:
        current_msa_paths = []
    if args.force or current_msa_paths != expected_msa_paths:
        chain_a, chain_b = system_name.split("_", 1) if "_" in system_name else (system_name, "B")
        write_af3_json(chain_a, chain_b, seq_a, seq_b, query_msa, msa_b, json_path,
                       absolute_msa_paths=True)
        # The shared writer derives the name from chain IDs. Keep a custom --system exact.
        if f"{chain_a}_{chain_b}" != system_name:
            data = json.loads(json_path.read_text())
            data["name"] = system_name
            json_path.write_text(json.dumps(data, indent=2) + "\n")
        LOG.info("Wrote %s", json_path)
    else:
        LOG.info("Keeping existing %s (use --force to rebuild)", json_path)

    return {
        "system_name": system_name,
        "af3_json": str(json_path),
        "af3_out_dir": str(root / "af3_predictions" / system_name),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fasta", type=Path, help="FASTA for one pair")
    inputs.add_argument("--fasta-single", type=Path, help="Query FASTA for batch mode")
    parser.add_argument("--msa-a", type=Path, help="Chain A A3M; optional in batch mode")
    parser.add_argument("--msa-b", type=Path, help="Chain B A3M in single-pair mode")
    parser.add_argument("--msa-dir", type=Path, help="Directory of chain B A3Ms in batch mode")
    parser.add_argument("--system", help="System name in single-pair mode (default: FASTA stem)")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Destination for AF3 inputs, query MSA, jobs.tsv and af3_cmds.txt")
    parser.add_argument("--tools", choices=("af3",), default="af3",
                        help="Compatibility option; this pipeline supports AF3 only")
    parser.add_argument("--force", action="store_true", help="Rebuild existing JSON and generated query MSA")
    parser.add_argument("--recreate-manifest", action="store_true", help="Replace existing jobs.tsv")
    parser.add_argument("--recreate-commands", action="store_true", help="Replace existing af3_cmds.txt")
    parser.add_argument("--max-identity", type=float, default=95.0)
    parser.add_argument("--no-filter-non-standard", action="store_true")
    parser.add_argument("--af3-num-recycles", type=int, default=1)
    parser.add_argument("--af3-num-diffusion-samples", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.af3_num_recycles < 0 or args.af3_num_diffusion_samples < 1:
        parser.error("AF3 recycles must be >= 0 and diffusion samples must be >= 1")
    if not 0 <= args.max_identity <= 100:
        parser.error("--max-identity must be between 0 and 100")
    if args.fasta and (not args.msa_a or not args.msa_b or args.msa_dir):
        parser.error("Single-pair mode requires --msa-a and --msa-b, without --msa-dir")
    if args.fasta_single and (not args.msa_dir or args.msa_b or args.system):
        parser.error("Batch mode requires --msa-dir, without --msa-b or --system")

    try:
        fasta = existing_file(args.fasta or args.fasta_single, "FASTA")
        msa_a = existing_file(args.msa_a, "Chain A MSA") if args.msa_a else None
        if args.fasta:
            pairs = [(args.system or fasta.stem, existing_file(args.msa_b, "Chain B MSA"))]
        else:
            msa_dir = args.msa_dir.resolve()
            if not msa_dir.is_dir():
                raise ValueError(f"MSA directory not found: {msa_dir}")
            pairs = [(f"{fasta.stem}_{path.stem}", path) for path in sorted(msa_dir.glob("*.a3m"))]
            if not pairs:
                raise ValueError(f"No .a3m files found in {msa_dir}")

        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "jobs.tsv"
        rows = load_manifest(manifest_path, args.recreate_manifest)
        prepared = 0
        args.excluded_systems = set()
        candidate_path = root / "screen_candidates.tsv"
        candidates = {}
        if candidate_path.is_file() and not args.recreate_manifest:
            with candidate_path.open(newline="") as handle:
                candidates = {r["system_name"]: r for r in csv.DictReader(handle, delimiter="\t")}
        for system_name, msa_b in pairs:
            candidate = dict(system_name=system_name, bait_accession=fasta.stem,
                             target_accession=msa_b.stem, chain_a_accession=fasta.stem,
                             chain_b_accession=msa_b.stem,
                             pair_key=json.dumps(sorted([fasta.stem, msa_b.stem])),
                             target_msa=str(msa_b.resolve()))
            row = prepare_pair(system_name, fasta, msa_a, msa_b, root, args, candidate)
            candidates[system_name] = candidate
            if row:
                rows[system_name] = row
                prepared += 1
            else:
                rows.pop(system_name, None)
                args.excluded_systems.add(system_name)
        with candidate_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidate), delimiter="\t")
            writer.writeheader()
            writer.writerows(candidates.values())
        save_manifest(manifest_path, rows, args.recreate_manifest)
        command_path = root / "af3_cmds.txt"
        command_count = save_commands(command_path, rows, args)
        LOG.info("Prepared %d AF3 pair(s); %d command(s), shortest first, in %s", prepared,
                 command_count, command_path)
        LOG.info("Chain and complex lengths: %s", root / "af3_lengths.csv")
        LOG.info("No inference was launched. Submit the generated commands with an AF3 runner.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
