import argparse
import csv
import shlex
from pathlib import Path
from typing import Optional


def extract_first_sequence(fasta_path: Path) -> str:
    """Extract the first sequence from a FASTA or A3M file.
    
    Args:
        fasta_path: Path to FASTA/A3M file
        
    Returns:
        First sequence as string (uppercase, no gaps)
    """
    seq_lines = []
    in_first_seq = False
    
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if in_first_seq:
                    break  # Found next header, stop
                in_first_seq = True
            elif in_first_seq:
                seq_lines.append(line)
    
    # Remove gaps and convert to uppercase
    sequence = ''.join(seq_lines).replace('-', '').upper()
    return sequence


def check_non_standard_residues(sequence: str) -> tuple[bool, str]:
    """Check if sequence contains non-standard amino acid residues.
    
    Standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    Non-standard codes: X, B, Z, J, O, U, *
    
    Args:
        sequence: Protein sequence (uppercase, single-letter codes)
        
    Returns:
        Tuple of (is_valid, non_standard_residues_found)
    """
    standard_residues = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Find non-standard residues
    non_standard = set(sequence) - standard_residues - set('-')
    
    if non_standard:
        residue_list = sorted(non_standard)
        return False, ''.join(residue_list)
    
    return True, ''


def parse_tools(tools_arg: str) -> set[str]:
    tools = {t.strip().lower() for t in tools_arg.split(",") if t.strip()}
    if "all" in tools:
        return {"af3", "boltz", "colabfold"}
    normalized = set()
    for tool in tools:
        if tool in {"af3", "alphafold3"}:
            normalized.add("af3")
        elif tool == "boltz":
            normalized.add("boltz")
        elif tool in {"colab", "colabfold"}:
            normalized.add("colabfold")
    return normalized


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: list[str], recreate: bool) -> None:
    ensure_parent(path)
    mode = "w" if recreate or not path.exists() else "a"
    if mode == "a":
        existing = set()
        with open(path, "r") as f:
            existing = {line.strip() for line in f if line.strip()}
        lines = [line for line in lines if line.strip() and line.strip() not in existing]

    with open(path, mode) as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def get_sequence_length(a3m_path: Path) -> float:
    try:
        with open(a3m_path, "r") as f:
            first_line = f.readline().strip()
            if first_line.startswith("#"):
                lengths = first_line[1:].split()[0].split(",")
                return float(sum(int(x) for x in lengths))
    except Exception:
        pass
    return float("inf")


def get_row_sort_key(row: dict[str, str]) -> tuple[float, str, str, str]:
    a3m = row.get("colab_a3m", "").strip()
    if a3m:
        length = get_sequence_length(Path(a3m))
    else:
        length = float("inf")
    af3_json = row.get("af3_json", "").strip()
    boltz_yaml = row.get("boltz_yaml", "").strip()
    job_name = row.get("job_name", "").strip()
    return (length, af3_json, boltz_yaml, job_name)


def get_row_colab_length(row: dict[str, str]) -> Optional[float]:
    a3m = row.get("colab_a3m", "").strip()
    if not a3m:
        return None
    return get_sequence_length(Path(a3m))

def load_colab_flags(flags_file: Path) -> list[str]:
    valid_flags = {
        "--msa-only", "--msa-mode", "--pair-mode", "--pair-strategy", "--templates",
        "--custom-template-path", "--max-template-date", "--max-template-hits",
        "--pdb-hit-file", "--local-pdb-path", "--num-recycle", "--recycle-early-stop-tolerance",
        "--num-ensemble", "--num-seeds", "--random-seed", "--num-models", "--model-type",
        "--model-order", "--initial-guess", "--use-dropout", "--max-seq", "--max-extra-seq",
        "--max-msa", "--disable-cluster-profile", "--calc-extra-ptm", "--no-use-probs-extra",
        "--data", "--amber", "--num-relax", "--relax-max-iterations", "--relax-tolerance",
        "--relax-stiffness", "--relax-max-outer-iterations", "--use-gpu-relax", "--rank",
        "--stop-at-score", "--jobname-prefix", "--save-all", "--save-recycles",
        "--save-single-representations", "--save-pair-representations", "--overwrite-existing-results",
        "--zip", "--sort-queries-by", "--host-url", "--disable-unified-memory",
        "--recompile-padding", "--debug-logging", "--af3-json"
    }

    if not flags_file.is_file():
        raise FileNotFoundError(f"Flags file not found: {flags_file}")

    flags = []
    with open(flags_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            flag = line.split()[0]
            if flag not in valid_flags:
                raise ValueError(f"Unknown flag '{flag}' on line {line_num} of {flags_file}")
            flags.append(line)
    return flags


def normalize_path_for_command(path_str: str, base_dir: Path, use_relative_paths: bool) -> str:
    """Normalize path for command emission.

    When use_relative_paths is enabled, absolute paths under base_dir are converted
    to relative paths so generated command files are portable within a screen folder.
    """
    path_str = path_str.strip()
    if not path_str:
        return path_str

    p = Path(path_str)
    if not use_relative_paths:
        return str(p)

    base_dir = base_dir.resolve()
    p_abs = p if p.is_absolute() else (base_dir / p)

    try:
        rel = p_abs.resolve().relative_to(base_dir)
        return str(rel)
    except Exception:
        # If path is outside base_dir, keep original representation.
        return str(p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AF3/Boltz/ColabFold command files from unified jobs.tsv"
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("jobs.tsv"),
        help="Path to unified jobs.tsv generated by run_pipeline.py",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="all",
        help="Tools to generate commands for: af3, boltz, colabfold, all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where command files are written",
    )
    parser.add_argument(
        "--flags-file",
        type=Path,
        default=Path("colab_config.txt"),
        help="File containing ColabFold flags",
    )
    parser.add_argument(
        "--length-threshold",
        type=int,
        default=1000,
        help="Length threshold for splitting ColabFold short/long commands",
    )
    parser.add_argument(
        "--skip-lengths",
        type=int,
        default=3000,
        help="Skip ColabFold commands with sequence length >= this value",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate command files instead of appending new commands",
    )
    parser.add_argument(
        "--af3-exec",
        type=str,
        default="run_alphafold.py",
        help="AF3 executable to use when generating AF3 commands",
    )
    parser.add_argument(
        "--af3-model-dir",
        type=str,
        default="/path/to/af3/weights",
        help="Model directory passed to AF3",
    )
    parser.add_argument(
        "--af3-db-dir",
        type=str,
        default="/path/to/af3/databases",
        help="Database directory passed to AF3",
    )
    parser.add_argument(
        "--af3-norun-data-pipeline",
        dest="af3_norun_data_pipeline",
        action="store_true",
        default=True,
        help="Pass --norun_data_pipeline to AF3 (default)",
    )
    parser.add_argument(
        "--af3-run-data-pipeline",
        dest="af3_norun_data_pipeline",
        action="store_false",
        help="Do not pass --norun_data_pipeline to AF3",
    )
    parser.add_argument(
        "--filter-non-standard",
        action="store_true",
        help="Filter out sequences with non-standard amino acid residues (X, B, Z, J, O, U, *)",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Keep absolute paths in generated commands (default: emit relative paths when possible)",
    )

    args = parser.parse_args()
    selected_tools = parse_tools(args.tools)
    use_relative_paths = not args.absolute_paths

    if not args.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_path}")

    rows = []
    with open(args.manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"No rows in manifest: {args.manifest_path}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_base_dir = args.output_dir.resolve()

    # Filter for non-standard residues if requested
    removed_non_standard = []
    if args.filter_non_standard:
        filtered_rows = []
        for row in rows:
            a3m = row.get('colab_a3m', '').strip()
            if not a3m or not Path(a3m).exists():
                # Keep rows with no A3M
                filtered_rows.append(row)
                continue
            
            try:
                seq = extract_first_sequence(Path(a3m))
                is_valid, non_std = check_non_standard_residues(seq)
                
                if is_valid:
                    filtered_rows.append(row)
                else:
                    system_name = row.get('system_name', 'unknown')
                    removed_non_standard.append((system_name, non_std))
                    print(f"⊘ Filtered: {system_name} (non-standard: {non_std})")
            except Exception as e:
                # Keep on error
                filtered_rows.append(row)
        
        rows = filtered_rows
        if removed_non_standard:
            removal_log = args.output_dir / "removed_non_standard.txt"
            with open(removal_log, 'w') as f:
                f.write(f"Filtered {len(removed_non_standard)} systems with non-standard residues:\n\n")
                for system_name, non_std_chars in removed_non_standard:
                    f.write(f"{system_name}\t{non_std_chars}\n")
            print(f"✓ Removal log: {removal_log}")

    sorted_rows = sorted(rows, key=get_row_sort_key)

    short_rows = []
    long_rows = []
    skipped_rows = []
    for row in sorted_rows:
        length = get_row_colab_length(row)
        if length is None:
            short_rows.append((row, length))
            continue
        if length >= args.skip_lengths:
            skipped_rows.append((row, length))
        elif length <= args.length_threshold:
            short_rows.append((row, length))
        else:
            long_rows.append((row, length))

    runnable_rows = [row for row, _ in short_rows] + [row for row, _ in long_rows]

    if "colabfold" in selected_tools:
        colab_flags = load_colab_flags(args.flags_file)
        colab_with_len = []
        for row, _ in short_rows + long_rows:
            a3m = row.get("colab_a3m", "").strip()
            out_dir = row.get("colab_out_dir", "").strip()
            if not a3m or not out_dir:
                continue
            a3m_path = Path(a3m)
            length = get_sequence_length(a3m_path)
            a3m_cmd = normalize_path_for_command(a3m, command_base_dir, use_relative_paths)
            out_dir_cmd = normalize_path_for_command(out_dir, command_base_dir, use_relative_paths)
            cmd_parts = ["colabfold_batch"] + colab_flags + [shlex.quote(a3m_cmd), shlex.quote(out_dir_cmd)]
            cmd = " ".join(cmd_parts)
            colab_with_len.append((length, cmd, a3m))

        short_cmds = []
        short_a3ms = []
        long_cmds = []
        long_a3ms = []
        skipped_a3ms = []

        for length, cmd, a3m in colab_with_len:
            a3m_name = Path(a3m).name
            if length <= args.length_threshold:
                short_cmds.append(cmd)
                short_a3ms.append(f"{a3m_name}\t{length}")
            else:
                long_cmds.append(cmd)
                long_a3ms.append(f"{a3m_name}\t{length}")

        for row, length in skipped_rows:
            a3m = row.get("colab_a3m", "").strip()
            if not a3m:
                continue
            a3m_name = Path(a3m).name
            skipped_a3ms.append(f"{a3m_name}\t{length}")

        all_cmds = short_cmds + long_cmds

        colab_file = args.output_dir / "colab_cmds.txt"
        short_a3m_file = args.output_dir / "short_a3ms.txt"
        long_a3m_file = args.output_dir / "long_a3ms.txt"
        skipped_a3m_file = args.output_dir / "skipped_a3ms.txt"

        write_lines(colab_file, all_cmds, args.recreate)
        write_lines(short_a3m_file, short_a3ms, args.recreate)
        write_lines(long_a3m_file, long_a3ms, args.recreate)
        write_lines(skipped_a3m_file, skipped_a3ms, args.recreate)

    if "af3" in selected_tools:
        af3_lines = []
        for row in runnable_rows:
            json_path = row.get("af3_json", "").strip()
            out_dir = row.get("af3_out_dir", "").strip()
            if not json_path or not out_dir:
                continue
            json_path_cmd = normalize_path_for_command(json_path, command_base_dir, use_relative_paths)
            out_dir_cmd = normalize_path_for_command(out_dir, command_base_dir, use_relative_paths)
            cmd_parts = [shlex.quote(args.af3_exec)]
            if args.af3_norun_data_pipeline:
                cmd_parts.append("--norun_data_pipeline")
            cmd_parts.extend(
                [
                    f"--json_path={shlex.quote(json_path_cmd)}",
                    f"--model_dir={shlex.quote(args.af3_model_dir)}",
                    f"--db_dir={shlex.quote(args.af3_db_dir)}",
                    # use the parent directory specified as af3 already uses the name of the folder lower case as the output name 
                    f"--output_dir={shlex.quote(str(Path(out_dir_cmd).parent))}",
                    f"--num_diffusion_samples 1",
                    f"--force_output_dir",
                ]
            )
            cmd = " ".join(cmd_parts)
            af3_lines.append(cmd)
        af3_file = args.output_dir / "af3_cmds.txt"
        write_lines(af3_file, af3_lines, args.recreate)
        print(f"AF3 commands: {len(af3_lines)} -> {af3_file}")

    if "boltz" in selected_tools:
        boltz_lines = []
        for row in runnable_rows:
            yaml_path = row.get("boltz_yaml", "").strip()
            out_dir = row.get("boltz_out_dir", "").strip()
            if not yaml_path or not out_dir:
                continue
            yaml_path_cmd = normalize_path_for_command(yaml_path, command_base_dir, use_relative_paths)
            out_dir_cmd = normalize_path_for_command(out_dir, command_base_dir, use_relative_paths)
            boltz_lines.append(
                f"boltz predict {shlex.quote(yaml_path_cmd)} --out_dir={shlex.quote(out_dir_cmd)}"
            )
        boltz_file = args.output_dir / "boltz_cmds.txt"
        write_lines(boltz_file, boltz_lines, args.recreate)
        print(f"Boltz commands: {len(boltz_lines)} -> {boltz_file}")

    kept_count = len(short_rows) + len(long_rows)
    print(
        f"kept systems: {kept_count} | short: {len(short_rows)} | long: {len(long_rows)} | skipped: {len(skipped_rows)} | non-standard: {len(removed_non_standard)}"
    )


if __name__ == "__main__":
    main()