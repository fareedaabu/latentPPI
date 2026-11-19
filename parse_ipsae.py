#!/usr/bin/env python3
import argparse
import csv
import fcntl
import json
import pickle
import re
import subprocess
import sys
import os
import tempfile
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path


SUMMARY_LOCK_SUFFIX = ".lock"
STATE_LOCK_SUFFIX = ".lock"


@dataclass(frozen=True)
class WorkItem:
    tool: str
    system: str
    rank: str
    model_num: str
    seed: str
    sample_num: str
    recycle: str
    pae_cutoff: str
    dist_cutoff: str
    pae_path: Path
    structure_path: Path
    output_name: str
    runlog_name: str
    extra_meta: dict[str, str]
    delete_pickle: Path | None = None
    convert_pickle: Path | None = None

    @property
    def key(self) -> str:
        parts = [
            self.tool,
            self.system,
            self.rank or "-",
            self.model_num or "-",
            self.seed or "-",
            self.sample_num or "-",
            self.recycle or "-",
            self.pae_cutoff,
            self.dist_cutoff,
        ]
        if self.extra_meta.get("attempt_id"):
            parts.append(self.extra_meta["attempt_id"])
        return "|".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover AF2/AF3 outputs, convert recycle pickles when needed, run ipsae.py, "
            "append summary rows safely, and track done/fail state."
        )
    )
    parser.add_argument("--root", default=".", help="Root directory to scan for AF2/AF3 outputs.")
    parser.add_argument("--outroot", help="Directory for ipSAE txt/log outputs.")
    parser.add_argument("--summary", help="Summary CSV path.")
    parser.add_argument("--failed-log", help="Failure log path.")
    parser.add_argument("--done-ledger", help="Persistent success ledger path.")
    parser.add_argument("--fail-ledger", help="Persistent failure ledger path.")
    parser.add_argument("--mode", choices=["auto", "af2", "af3"], default="auto", help="Discovery mode.")
    parser.add_argument("--pae", type=int, default=15, help="PAE cutoff passed to ipsae.py.")
    parser.add_argument("--dist", type=int, default=15, help="Distance cutoff passed to ipsae.py.")
    parser.add_argument("--ipsae-py", default=None, help="Path to ipsae.py. Defaults next to this script.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for ipsae.py runs.")
    parser.add_argument("--ignore-af3-seed-samples",action="store_true", help="Ignore AF3 seed-sample prediction folders during discovery.")
    parser.add_argument("--prefer-seed-samples", action="store_true",
                        help="Score seed/sample structures when present, avoiding duplicate top-ranked copies.")
    parser.add_argument("--input-json", type=Path, help="Original AF3 input, for exact name and single-seed fallback.")
    parser.add_argument("--attempt-id", default=os.environ.get("LATENTPPI_ATTEMPT_ID", ""))
    parser.add_argument("--discovery-only", action="store_true", help="List discovered work items without running ipsae.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of work items to process.")
    parser.add_argument("--verbose", action="store_true", help="Print per-item progress.")
    return parser.parse_args()


def format_cutoff(value: int) -> str:
    return str(value)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class LockedFile:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        ensure_parent(self.path)
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        self.handle.seek(0)
        return self.handle

    def __exit__(self, exc_type, exc, tb):
        assert self.handle is not None
        self.handle.flush()
        self.handle.close()


def load_ledger(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_ledger(path: Path, key: str) -> None:
    lock_path = path.with_name(path.name + STATE_LOCK_SUFFIX)
    with LockedFile(lock_path):
        existing = load_ledger(path)
        if key in existing:
            return
        ensure_parent(path)
        with path.open("a") as handle:
            handle.write(f"{key}\n")


def log_failure(path: Path, key: str, reason: str) -> None:
    lock_path = path.with_name(path.name + STATE_LOCK_SUFFIX)
    with LockedFile(lock_path):
        seen_keys = set()
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                seen_keys.add(line.split("\t", 1)[0])
        if key in seen_keys:
            return
        ensure_parent(path)
        with path.open("a") as handle:
            handle.write(f"{key}\t{reason}\n")


def append_text_log(path: Path, line: str) -> None:
    ensure_parent(path)
    with path.open("a") as handle:
        handle.write(line.rstrip("\n") + "\n")


def convert_pickle_to_json(pickle_file: Path, out_file: Path) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to convert recycle pickles") from exc

    with pickle_file.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{pickle_file} did not contain a dict-like ColabFold result object")

    out: dict[str, object] = {}

    if "plddt" in data:
        plddt = np.asarray(data["plddt"])
        out["plddt"] = np.around(plddt.astype(float), 2).tolist()

    if "predicted_aligned_error" in data:
        pae = np.asarray(data["predicted_aligned_error"])
        out["predicted_aligned_error"] = np.around(pae.astype(float), 2).tolist()
        out["max_pae"] = float(np.max(pae))
    elif "pae" in data:
        pae = np.asarray(data["pae"])
        out["pae"] = np.around(pae.astype(float), 2).tolist()
        out["max_pae"] = float(np.max(pae))
    else:
        raise ValueError(
            f"{pickle_file} does not contain 'predicted_aligned_error' or 'pae'; cannot score recycle model"
        )

    for key in ["iptm", "ptm", "ranking_confidence", "mean_plddt", "tol", "actifptm"]:
        if key not in data:
            continue
        try:
            out[key] = float(data[key])
        except Exception:
            continue

    stem = pickle_file.stem
    recycle_tag = stem.rsplit(".", 1)[-1]
    if recycle_tag.startswith("r") and recycle_tag[1:].isdigit():
        out["recycle"] = int(recycle_tag[1:])

    ensure_parent(out_file)
    with out_file.open("w") as handle:
        json.dump(out, handle)


def default_paths(root: Path, mode: str) -> tuple[Path, Path]:
    if mode == "af3":
        summary_root = root.parent
        return root.parent / "ipsae_scores_af3", summary_root / "ipsae_summary_af3.csv"
    return root / "ipsae_scores", root / "ipsae_scores" / "ipsae_summary.csv"


AF3_SEED_SAMPLE_DIR_PATTERN = re.compile(r"^seed[-_]?(\d+)[_-]sample[-_]?(\d+)$", flags=re.IGNORECASE)


def build_af2_final_item(json_path: Path, pae_str: str, dist_str: str) -> WorkItem | None:
    json_name = json_path.name
    marker = "_scores_rank_"
    if marker not in json_name:
        return None
    system, tail = json_name.split(marker, 1)
    if not tail.endswith(".json"):
        return None
    parts = tail[:-5].split("_")
    if len(parts) < 8:
        return None
    rank, _, _, _, _, model_num, _, seed = parts[:8]
    structure_name = json_name.replace("_scores_", "_unrelaxed_").replace(".json", ".pdb")
    structure_path = json_path.with_name(structure_name)
    if not structure_path.exists():
        return None
    return WorkItem(
        tool="af2",
        system=system,
        rank=rank,
        model_num=model_num,
        seed=seed,
        sample_num="",
        recycle="",
        pae_cutoff=pae_str,
        dist_cutoff=dist_str,
        pae_path=json_path,
        structure_path=structure_path,
        output_name=f"{structure_path.stem}_{pae_str}_{dist_str}.txt",
        runlog_name=f"{structure_path.stem}_ipsae_run.log",
        extra_meta={
            "ipsae_file": "",
            "system": system,
            "rank": rank,
            "model_num": model_num,
            "seed": seed,
            "sample_num": "",
            "recycle": "",
            "recycle_tag": "",
            "pae_cutoff": pae_str,
            "dist_cutoff": dist_str,
        },
    )


def build_af2_recycle_item(pickle_path: Path, pae_str: str, dist_str: str) -> WorkItem | None:
    if ".r" not in pickle_path.stem:
        return None
    basename = pickle_path.name
    marker = "_all_rank_"
    if marker not in basename:
        return None
    system, tail = basename.split(marker, 1)
    if not tail.endswith(".pickle"):
        return None
    tail_no_ext = tail[:-7]
    parts = tail_no_ext.split(".")
    if len(parts) != 2 or not parts[1].startswith("r"):
        return None
    recycle_tag = parts[1]
    fields = parts[0].split("_")
    if len(fields) < 8:
        return None
    rank, _, _, _, _, model_num, _, seed = fields[:8]
    structure_name = (
        f"{system}_unrelaxed_rank_{rank}_alphafold2_multimer_v3_model_{model_num}_seed_{seed}.{recycle_tag}.pdb"
    )
    structure_path = pickle_path.with_name(structure_name)
    if not structure_path.exists():
        return None
    pae_json = pickle_path.with_suffix(".json")
    return WorkItem(
        tool="af2",
        system=system,
        rank=rank,
        model_num=model_num,
        seed=seed,
        sample_num="",
        recycle=recycle_tag[1:],
        pae_cutoff=pae_str,
        dist_cutoff=dist_str,
        pae_path=pae_json,
        structure_path=structure_path,
        output_name=f"{structure_path.stem}_{pae_str}_{dist_str}.txt",
        runlog_name=f"{structure_path.stem}_ipsae_run.log",
        extra_meta={
            "ipsae_file": "",
            "system": system,
            "rank": rank,
            "model_num": model_num,
            "seed": seed,
            "sample_num": "",
            "recycle": recycle_tag[1:],
            "recycle_tag": recycle_tag,
            "pae_cutoff": pae_str,
            "dist_cutoff": dist_str,
        },
        delete_pickle=pickle_path,
        convert_pickle=pickle_path,
    )


def discover_af2(root: Path, pae_str: str, dist_str: str) -> list[WorkItem]:
    items: list[WorkItem] = []
    seen_keys: set[str] = set()

    for json_path in sorted(root.rglob("*_scores_rank_*_alphafold2_multimer_v3_model_*_seed_*.json")):
        item = build_af2_final_item(json_path, pae_str, dist_str)
        if item and item.key not in seen_keys:
            items.append(item)
            seen_keys.add(item.key)

    for pickle_path in sorted(root.rglob("*_all_rank_*_alphafold2_multimer_v3_model_*_seed_*.r*.pickle")):
        item = build_af2_recycle_item(pickle_path, pae_str, dist_str)
        if item and item.key not in seen_keys:
            items.append(item)
            seen_keys.add(item.key)

    return items


def af3_system_name_from_path(json_path: Path, fallback: str) -> str:
    parent = json_path.parent.name
    grandparent = json_path.parent.parent.name

    match = AF3_SEED_SAMPLE_DIR_PATTERN.match(parent)
    if match and grandparent:
        return grandparent

    # Accept both AF3 layouts:
    # 1) <System>/<system_lower>/confidences.json
    # 2) <System>/<System>/confidences.json (or same-case variants)
    if parent and grandparent and parent.lower() == grandparent.lower():
        return grandparent

    return parent or fallback


def af3_seed_sample_from_path(json_path: Path) -> tuple[str, str]:
    match = AF3_SEED_SAMPLE_DIR_PATTERN.match(json_path.parent.name)
    if match:
        return match.group(1), match.group(2)
    return "", ""


def is_af3_seed_sample_prediction(json_path: Path) -> bool:
    return (json_path.name == "confidences.json" or json_path.name.endswith("_confidences.json")) and bool(
        AF3_SEED_SAMPLE_DIR_PATTERN.match(json_path.parent.name))


def build_af3_item(
    json_path: Path,
    pae_str: str,
    dist_str: str,
    ignore_seed_samples: bool = False,
) -> WorkItem | None:
    if ignore_seed_samples and is_af3_seed_sample_prediction(json_path):
        return None

    name = json_path.name
    if name == "summary_confidences.json" or name.endswith("_summary_confidences.json"):
        return None
    if name == "confidences.json":
        structure_path = json_path.with_name("model.cif")
        stem = af3_system_name_from_path(json_path, json_path.parent.name)
        model_num = ""
        seed, sample_num = af3_seed_sample_from_path(json_path)
    elif name.endswith("_confidences.json"):
        structure_path = json_path.with_name(name.replace("_confidences.json", "_model.cif"))
        stem = name[:-len("_confidences.json")]
        model_num = stem.rsplit("_model_", 1)[-1] if "_model_" in stem else ""
        seed, sample_num = af3_seed_sample_from_path(json_path)
        if seed:
            stem = json_path.parent.parent.name
    else:
        return None
    if not structure_path.exists():
        return None
    return WorkItem(
        tool="af3",
        system=stem,
        rank="",
        model_num=model_num,
        seed=seed,
        sample_num=sample_num,
        recycle="",
        pae_cutoff=pae_str,
        dist_cutoff=dist_str,
        pae_path=json_path,
        structure_path=structure_path,
        output_name=f"{structure_path.stem}_{pae_str}_{dist_str}.txt",
        runlog_name=f"{structure_path.stem}_ipsae_run.log",
        extra_meta={
            "ipsae_file": "",
            "af3_json": str(json_path),
            "af3_cif": str(structure_path),
            "system": stem,
            "model_num": model_num,
            "seed": seed,
            "sample_num": sample_num,
            "pae_cutoff": pae_str,
            "dist_cutoff": dist_str,
        },
    )


def discover_af3(root: Path, pae_str: str, dist_str: str, ignore_seed_samples: bool = False) -> list[WorkItem]:
    items: list[WorkItem] = []
    for json_path in sorted(root.rglob("*confidences.json")):
        item = build_af3_item(json_path, pae_str, dist_str, ignore_seed_samples=ignore_seed_samples)
        if item is not None:
            items.append(item)
    return items


def parse_table(txt_path: Path) -> tuple[list[str], list[list[str]]]:
    lines = []
    for raw_line in txt_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    if not lines:
        raise ValueError(f"empty table: {txt_path}")
    header = lines[0].split()
    rows = [line.split() for line in lines[1:]]
    return header, rows


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def plddt_from_json(json_path: Path) -> str:
    try:
        with json_path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
    except Exception:
        return ""

    if not isinstance(data, dict):
        return ""

    mean_plddt = data.get("mean_plddt")
    if isinstance(mean_plddt, (int, float)):
        return f"{float(mean_plddt):.2f}"

    for key in ["plddt", "plDDT", "atom_plddts"]:
        values = data.get(key)
        if not isinstance(values, list):
            continue
        numeric_values: list[float] = []
        for value in values:
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
        mean_value = _safe_mean(numeric_values)
        if mean_value is not None:
            return f"{mean_value:.2f}"

    return ""


def plddt_from_structure(structure_path: Path) -> str:
    suffix = structure_path.suffix.lower()
    values: list[float] = []

    try:
        with structure_path.open("r", encoding="utf-8", errors="replace") as handle:
            if suffix == ".cif":
                for line in handle:
                    line = line.strip()
                    if not line or not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                    fields = line.split()
                    if len(fields) < 15:
                        continue
                    try:
                        values.append(float(fields[14]))
                    except ValueError:
                        continue
            elif suffix == ".pdb":
                for raw_line in handle:
                    if not (raw_line.startswith("ATOM") or raw_line.startswith("HETATM")):
                        continue
                    if len(raw_line) < 66:
                        continue
                    try:
                        values.append(float(raw_line[60:66].strip()))
                    except ValueError:
                        continue
    except Exception:
        return ""

    mean_value = _safe_mean(values)
    if mean_value is None:
        return ""
    return f"{mean_value:.2f}"


def extract_plddt(item: WorkItem) -> str:
    plddt = plddt_from_json(item.pae_path)
    if plddt:
        return plddt
    return plddt_from_structure(item.structure_path)


def append_summary(summary_path: Path, item: WorkItem, txt_path: Path) -> None:
    header, rows = parse_table(txt_path)
    meta = dict(item.extra_meta)
    meta["ipsae_file"] = str(txt_path)
    meta["plddt"] = extract_plddt(item)
    legacy_meta = {k: v for k, v in meta.items() if k != "plddt"}
    lock_path = summary_path.with_name(summary_path.name + SUMMARY_LOCK_SUFFIX)

    with LockedFile(lock_path):
        existing_header = None
        if summary_path.exists() and summary_path.stat().st_size > 0:
            with summary_path.open(newline="") as handle:
                reader = csv.reader(handle)
                existing_header = next(reader, None)

        expected_header = list(meta.keys()) + header
        legacy_header = list(legacy_meta.keys()) + header
        if existing_header is None:
            row_meta = meta
        else:
            if len(existing_header) < len(header) or existing_header[-len(header):] != header:
                raise ValueError(f"summary header mismatch in {summary_path}")

            existing_meta_keys = existing_header[:-len(header)]
            missing = [key for key in meta if key not in existing_meta_keys]
            if missing:
                # Preserve older rows, leaving new provenance columns blank for them.
                with summary_path.open(newline="") as handle:
                    old_rows = list(csv.DictReader(handle))
                existing_meta_keys += missing
                with tempfile.NamedTemporaryFile(mode="w", dir=summary_path.parent,
                        prefix=".ipsae-", newline="", delete=False) as handle:
                    temporary = Path(handle.name)
                    writer = csv.DictWriter(handle, fieldnames=existing_meta_keys + header)
                    writer.writeheader()
                    writer.writerows(old_rows)
                temporary.replace(summary_path)
            row_meta = {key: meta.get(key, "") for key in existing_meta_keys}

        ensure_parent(summary_path)
        with summary_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if existing_header is None:
                writer.writerow(expected_header)
            for row in rows:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[:len(header)]
                writer.writerow([row_meta[key] for key in row_meta.keys()] + row)


def run_ipsae(python_exe: str, ipsae_py: Path, item: WorkItem, runlog_path: Path) -> subprocess.CompletedProcess[str]:
    ensure_parent(runlog_path)
    command = [
        python_exe,
        str(ipsae_py),
        str(item.pae_path),
        str(item.structure_path),
        item.pae_cutoff,
        item.dist_cutoff,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    with runlog_path.open("w") as handle:
        if result.stdout:
            handle.write(result.stdout)
        if result.stderr:
            handle.write(result.stderr)
    return result


def process_item(
    item: WorkItem,
    outroot: Path,
    summary_path: Path,
    done_ledger: Path,
    fail_ledger: Path,
    failed_log: Path,
    ipsae_py: Path,
    python_exe: str,
    verbose: bool,
) -> tuple[str, str]:
    if item.key in load_ledger(done_ledger):
        return "skip", item.key

    prefix = f"{item.system}_seed-{item.seed}_sample-{item.sample_num}_" if item.sample_num else ""
    final_out = outroot / (prefix + item.output_name)
    runlog_path = outroot / (prefix + item.runlog_name)

    try:
        if item.convert_pickle is not None and not item.pae_path.exists():
            convert_pickle_to_json(item.convert_pickle, item.pae_path)

        if not item.pae_path.exists():
            raise FileNotFoundError(f"missing PAE input: {item.pae_path}")
        if not item.structure_path.exists():
            raise FileNotFoundError(f"missing structure input: {item.structure_path}")

        result = run_ipsae(python_exe, ipsae_py, item, runlog_path)
        if result.returncode != 0:
            raise RuntimeError(f"ipsae exited with {result.returncode}; see {runlog_path}")

        expected_out = item.structure_path.with_name(item.output_name)
        if not expected_out.exists():
            raise FileNotFoundError(f"expected ipSAE output not found: {expected_out}")

        ensure_parent(final_out)
        expected_out.replace(final_out)
        append_summary(summary_path, item, final_out)
        append_ledger(done_ledger, item.key)

        if item.delete_pickle is not None and item.delete_pickle.exists():
            item.delete_pickle.unlink()

        if verbose:
            print(f"[OK] {item.key} -> {final_out}")
        return "ok", item.key
    except Exception as exc:
        reason = str(exc)
        log_failure(fail_ledger, item.key, reason)
        append_text_log(failed_log, f"{item.key}\t{reason}")
        if verbose:
            print(f"[FAIL] {item.key}: {reason}")
        return "fail", item.key


def select_mode(mode: str, root: Path) -> str:
    if mode != "auto":
        return mode
    if any(root.rglob("*confidences.json")):
        return "af3"
    return "af2"


def discover_items(
    root: Path,
    mode: str,
    pae_str: str,
    dist_str: str,
    ignore_af3_seed_samples: bool = False,
) -> list[WorkItem]:
    if mode == "af3":
        return discover_af3(root, pae_str, dist_str, ignore_seed_samples=ignore_af3_seed_samples)
    return discover_af2(root, pae_str, dist_str)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    mode = select_mode(args.mode, root)
    pae_str = format_cutoff(args.pae)
    dist_str = format_cutoff(args.dist)
    default_outroot, default_summary = default_paths(root, mode)
    outroot = Path(args.outroot).resolve() if args.outroot else default_outroot.resolve()
    summary_path = Path(args.summary).resolve() if args.summary else default_summary.resolve()
    failed_log = Path(args.failed_log).resolve() if args.failed_log else (outroot / "failed.txt")
    done_ledger = Path(args.done_ledger).resolve() if args.done_ledger else (outroot / "done_ledger.txt")
    fail_ledger = Path(args.fail_ledger).resolve() if args.fail_ledger else (outroot / "fail_ledger.txt")
    ipsae_py = Path(args.ipsae_py).resolve() if args.ipsae_py else (Path(__file__).resolve().parent / "ipsae.py")

    items = discover_items(
        root,
        mode,
        pae_str,
        dist_str,
        ignore_af3_seed_samples=args.ignore_af3_seed_samples,
    )
    if args.prefer_seed_samples and any(item.sample_num for item in items):
        items = [item for item in items if item.sample_num]
    source = json.loads(args.input_json.read_text()) if args.input_json else {}
    seeds = source.get("modelSeeds", [])
    if args.attempt_id and source:
        from record_run_env import fresh_structure
        attempt_path = root.parent / ".latentppi_attempts" / (hashlib.sha256(source["name"].encode()).hexdigest() + ".json")
        attempt = json.loads(attempt_path.read_text())
        if attempt.get("attempt_id") != args.attempt_id:
            raise ValueError("ipSAE attempt does not match the current inference attempt")
        items = [item for item in items if fresh_structure(item.structure_path, attempt)]
        if not items:
            raise ValueError("No fresh structure found for the current attempt")
    prepared = []
    for item in items:
        seed = item.seed or (str(seeds[0]) if len(seeds) == 1 else "")
        system = source.get("name", item.system)
        meta = {**item.extra_meta, "system": system, "seed": seed,
                "attempt_id": args.attempt_id,
                "requested_recycles": os.environ.get("LATENTPPI_REQUESTED_RECYCLES", "")}
        prepared.append(replace(item, seed=seed, system=system, extra_meta=meta))
    items = prepared
    if args.limit is not None:
        items = items[: args.limit]

    print(f"mode={mode} root={root}")
    print(f"discovered={len(items)} outroot={outroot} summary={summary_path}")

    if args.discovery_only:
        for item in items:
            print(
                f"DISCOVERED\t{item.key}\tpae={item.pae_path}\tstructure={item.structure_path}\tdelete_pickle={item.delete_pickle or ''}"
            )
        return 0

    ok_count = 0
    skip_count = 0
    fail_count = 0
    for item in items:
        status, _ = process_item(
            item=item,
            outroot=outroot,
            summary_path=summary_path,
            done_ledger=done_ledger,
            fail_ledger=fail_ledger,
            failed_log=failed_log,
            ipsae_py=ipsae_py,
            python_exe=args.python,
            verbose=args.verbose,
        )
        if status == "ok":
            ok_count += 1
        elif status == "skip":
            skip_count += 1
        else:
            fail_count += 1

    print(f"completed ok={ok_count} skip={skip_count} fail={fail_count}")
    print(f"done_ledger={done_ledger}")
    print(f"fail_ledger={fail_ledger}")
    print(f"failed_log={failed_log}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
