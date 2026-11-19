#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def relpath_from_manifest(path: Path, manifest_path: Path) -> str:
    """Return path relative to the manifest location when possible."""
    try:
        return str(path.resolve().relative_to(manifest_path.parent.resolve()))
    except Exception:
        return str(path)


def build_system_name(row: dict[str, str]) -> str:
    a = row.get("uniprot_a", "").strip()
    b = row.get("uniprot_b", "").strip()
    if not a or not b:
        return ""
    return f"{a}_{b}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert summary TSV (uniprot_a/uniprot_b) to a manifest usable by gen_commands.py"
    )
    parser.add_argument(
        "--input-tsv",
        type=Path,
        required=True,
        help="Input summary TSV with at least uniprot_a and uniprot_b columns",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("dataset_build/ppi_dataset/jobs.tsv"),
        help="Output jobs manifest path",
    )
    parser.add_argument(
        "--af3-json-dir",
        type=Path,
        default=Path("dataset_build/ppi_dataset/inputs/jsons"),
        help="Directory containing AF3 input JSON files named {uniprot_a}_{uniprot_b}.json",
    )
    parser.add_argument(
        "--af3-out-base",
        type=Path,
        default=Path("af3_predictions"),
        help="Base AF3 output directory; af3_out_dir becomes {base}/{system_name}",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths instead of paths relative to output manifest",
    )
    parser.add_argument(
        "--include-missing-json",
        action="store_true",
        help="Include rows even when expected AF3 JSON is missing",
    )
    args = parser.parse_args()

    if not args.input_tsv.is_file():
        raise FileNotFoundError(f"Input TSV not found: {args.input_tsv}")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_tsv, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = set(reader.fieldnames or [])
        required = {"uniprot_a", "uniprot_b"}
        missing = required - cols
        if missing:
            raise ValueError(f"Input TSV missing required columns: {sorted(missing)}")

        rows_out: list[dict[str, str]] = []
        total = 0
        skipped_missing_json = 0

        for row in reader:
            total += 1
            system_name = build_system_name(row)
            if not system_name:
                continue

            af3_json = args.af3_json_dir / f"{system_name}.json"
            af3_out = args.af3_out_base / system_name

            if not af3_json.exists() and not args.include_missing_json:
                skipped_missing_json += 1
                continue

            if args.absolute_paths:
                af3_json_str = str(af3_json.resolve())
                af3_out_str = str(af3_out.resolve())
            else:
                af3_json_str = relpath_from_manifest(af3_json, args.output_manifest)
                af3_out_str = relpath_from_manifest(af3_out, args.output_manifest)

            rows_out.append(
                {
                    "job_name": system_name,
                    "system_name": system_name,
                    "af3_json": af3_json_str,
                    "af3_out_dir": af3_out_str,
                }
            )

    with open(args.output_manifest, "w", newline="") as f:
        fieldnames = ["job_name", "system_name", "af3_json", "af3_out_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)

    print(
        f"Wrote {len(rows_out)} rows to {args.output_manifest} "
        f"(input rows: {total}, skipped missing json: {skipped_missing_json})"
    )


if __name__ == "__main__":
    main()
