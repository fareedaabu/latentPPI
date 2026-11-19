#!/usr/bin/env python3
"""Collect append-only AF3 timings; see METRICS.md for schema and selection rules."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import hashlib
import fcntl

import pandas as pd


ENV_FIELDS = ("gpu_type", "node", "slurm_job_id", "slurm_array_task_id")
BASE_FIELDS = (
    "system_name", "seed", "line_number", "chain_a_length", "chain_b_length",
    "total_length", "af3_json", "af3_out_dir", "json_path", "status", "gated",
    "total_seconds", "recycle_timing_consistent", "source_file", "source_line",
    *ENV_FIELDS,
)
PRED_FIELDS = {"latentppi_scorer_pred_metrics", "latentppi_scorer_pred_metric",
               "latentppi_scorer_pred_metric_name"}


def finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def seed_key(value):
    """Normalize integer seeds without passing large integers through float."""
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        return None
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if isinstance(value, float) and value != result:
        return None
    return str(result) if result >= 0 else None


def parse_record(line):
    def reject_constant(token):
        raise ValueError(token)

    try:
        return json.loads(line, parse_constant=reject_constant), False
    except ValueError:
        return json.loads(line, parse_constant=lambda _: None), True


def read_manifest(path, delimiter):
    if path is None:
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if "system_name" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing system_name column")
        rows = {}
        for row in reader:
            name = row.get("system_name")
            if not name or name in rows:
                raise ValueError(f"{path}: empty or duplicate system_name: {name!r}")
            rows[name] = {k: v for k, v in row.items() if k and v != ""}
        return rows


def incomplete(record):
    trunk = finite_number(record.get("trunk_time_seconds"))
    return (finite_number(record.get("latentppi_chain_len_a")) == 0
            and trunk is not None and 0 <= trunk < 0.001)


def derive_record(record):
    trunk = finite_number(record.get("trunk_time_seconds"))
    diffusion = finite_number(record.get("diffusion_time_seconds"))
    recycle = record.get("trunk_per_recycle_seconds")
    valid_recycle = (isinstance(recycle, list) and bool(recycle)
                     and all(finite_number(x) is not None and finite_number(x) >= 0
                             for x in recycle))
    times_ok = trunk is not None and diffusion is not None and trunk >= 0 and diffusion >= 0
    status = "ok" if times_ok and valid_recycle else "invalid_timing"
    if any(k not in record for k in ("trunk_time_seconds", "diffusion_time_seconds",
                                     "trunk_per_recycle_seconds")):
        status = "missing_timing"
    consistent = None
    if trunk is not None and trunk >= 0 and valid_recycle:
        consistent = abs(sum(float(x) for x in recycle) - trunk) <= 0.2 * trunk
    return {
        "status": status,
        "gated": (record.get("latentppi_scorer_gate_triggered") is True
                  and record.get("latentppi_scorer_gate_status") == "ok"),
        "total_seconds": trunk + diffusion if times_ok else None,
        "recycle_timing_consistent": consistent,
    }


def select_ipsae_rows(frame, rule):
    """Select an explicit whole row per (system, seed), rejecting unresolved ties.

    'max_ipsae' uses Type=max and the highest ipSAE (the owner-selected default).
    Equal ipSAE values prefer lower model/sample numbers, then lexical row content.
    Other rules: 'unique' or JSON {"where": {column: value}, "order_by": [...]}.
    Filters use exact values (null matches blanks); ordering is ascending, with a
    '-' prefix for descending. All sort-key ties are errors, never file-order wins.
    JSON rules have no implicit Type, direction, model, sample, cutoff, or ranking.
    """
    if not rule:
        raise ValueError("--ipsae-row-select is required when joining an ipSAE summary")
    if rule == "max_ipsae":
        if not {"Type", "ipSAE"}.issubset(frame.columns):
            raise ValueError("max_ipsae selection requires Type and ipSAE columns")
        selected = frame.loc[frame["Type"].eq("max")].copy()
        selected["ipSAE"] = selected["ipSAE"].map(finite_number)
        selected = selected.loc[selected["ipSAE"].notna()]
        if selected.empty:
            return selected
        order, ascending = ["ipSAE"], [False]
        temporary = []
        for column in ("model_num", "sample_num"):
            if column in selected:
                key = "_select_" + column
                selected[key] = pd.to_numeric(selected[column], errors="coerce")
                order.append(key)
                ascending.append(True)
                temporary.append(key)
        selected["_select_tie"] = selected.apply(
            lambda row: json.dumps({k: str(row[k]) for k in sorted(frame.columns)}, sort_keys=True),
            axis=1)
        selected = selected.sort_values(order + ["_select_tie"], ascending=ascending + [True],
                                        na_position="last", kind="stable")
        return selected.drop_duplicates(["system", "seed"]).drop(columns=temporary + ["_select_tie"])
    try:
        spec = {} if rule == "unique" else json.loads(rule)
    except ValueError as exc:
        raise ValueError("--ipsae-row-select must be 'unique' or a JSON rule") from exc
    if not isinstance(spec, dict) or set(spec) - {"where", "order_by"}:
        raise ValueError("ipSAE rule must contain only 'where' and/or 'order_by'")
    where, order = spec.get("where", {}), spec.get("order_by", [])
    if not isinstance(where, dict) or not isinstance(order, list):
        raise ValueError("ipSAE 'where' must be an object and 'order_by' a list")
    selected = frame.copy()
    for column, value in where.items():
        if column not in selected:
            raise ValueError(f"ipSAE filter column absent: {column}")
        series = selected[column]
        if value is None:
            mask = series.isna() | series.eq("")
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            mask = pd.to_numeric(series, errors="coerce").eq(value)
        elif isinstance(value, str):
            mask = series.astype(str).eq(value)
        else:
            raise ValueError(f"ipSAE filter for {column} must be a string, number, or null")
        selected = selected.loc[mask]
    keys = ["system", "seed"]
    columns, ascending = [], []
    for entry in order:
        if not isinstance(entry, str) or not entry.lstrip("-"):
            raise ValueError("ipSAE order_by entries must be column names")
        column = entry[1:] if entry.startswith("-") else entry
        if column not in selected or column in keys or column in columns:
            raise ValueError(f"invalid ipSAE order_by column: {column}")
        columns.append(column)
        ascending.append(not entry.startswith("-"))
        values = selected[column].replace("", None)
        numeric = pd.to_numeric(values, errors="coerce")
        selected[column] = numeric if numeric.notna().sum() == values.notna().sum() else values
    if columns:
        selected = selected.sort_values(columns, ascending=ascending, na_position="last",
                                        kind="stable")
        # Only ties at the chosen position matter. Ties among losing rows are harmless.
        winners = selected.drop_duplicates(keys)
        ties = selected.merge(winners[keys + columns], on=keys + columns, how="inner")
    else:
        ties = selected
    ambiguous = ties.duplicated(keys, keep=False)
    if ambiguous.any():
        examples = ties.loc[ambiguous, keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"ipSAE row selection is ambiguous for {examples}; refine the rule")
    return selected.drop_duplicates(keys)


def load_attempts(root):
    attempts = {}
    for path in (Path(root) / ".latentppi_attempts").glob("*.json"):
        attempt = json.loads(path.read_text())
        attempts[attempt["system_name"]] = attempt
    return attempts


def collect(root, jobs_path=None, lengths_path=None, pattern="**/prediction_timings.jsonl", attempts=None):
    attempts = load_attempts(root) if attempts is None else attempts
    jobs = read_manifest(jobs_path, "\t")
    lengths = read_manifest(lengths_path, ",")
    manifests = {name: {**jobs.get(name, {}), **lengths.get(name, {})}
                 for name in jobs.keys() | lengths.keys()}
    canonical = {}
    for name in manifests.keys() | attempts.keys():
        if name.casefold() in canonical and canonical[name.casefold()] != name:
            raise ValueError(f"Ambiguous system names differing only in case: {name}")
        canonical[name.casefold()] = name
    # Use manifest paths for failed/empty files; never infer an ID by splitting a name.
    folder_names = {}
    for name, job in jobs.items():
        if job.get("af3_out_dir"):
            folder = Path(job["af3_out_dir"])
            if not folder.is_absolute():
                folder = jobs_path.parent / folder
            folder = folder.resolve()
            if folder in folder_names and folder_names[folder] != name:
                raise ValueError(f"ambiguous af3_out_dir in jobs.tsv: {folder}")
            folder_names[folder] = name
    stats = Counter()
    metric_names, records, counts, failed_systems = set(), {}, Counter(), {}
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    stats["files_found"] = len(files)
    for path in files:
        resolved = path.resolve()
        owner = next((folder_names[p] for p in resolved.parents if p in folder_names), None)
        if owner is None:
            owner = next((canonical[p.name.casefold()] for p in resolved.parents
                          if p.name.casefold() in canonical), None)
        attempt = attempts.get(owner, {})
        skip_lines = 0
        baseline = attempt.get("timing_baselines", {}).get(str(resolved))
        if baseline:
            with path.open("rb") as handle:
                prefix = handle.read(baseline["size"])
            stat = path.stat()
            modified = ([stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != baseline["signature"]
                        if "signature" in baseline else
                        stat.st_mtime >= attempt.get("started_at", float("inf")))
            rewritten = stat.st_size <= baseline["size"] and modified
            if not rewritten and hashlib.sha256(prefix).hexdigest() == baseline["sha256"]:
                skip_lines = baseline["lines"]
        sidecar = {}
        env_path = path.parent / "run_env.json"
        if env_path.is_file():
            try:
                env, _ = parse_record(env_path.read_text())
                if not isinstance(env, dict):
                    raise ValueError("expected object")
                sidecar = {k: env.get(k) for k in ENV_FIELDS}
            except (OSError, ValueError):
                stats["invalid_sidecars"] += 1
        parsed_file = False
        file_reason = "empty_file"
        try:
            with path.open(errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line_number <= skip_lines:
                        continue
                    if not line.strip():
                        continue
                    try:
                        record, lenient = parse_record(line)
                        if not isinstance(record, dict):
                            raise ValueError("expected object")
                    except ValueError:
                        stats["invalid_lines"] += 1
                        file_reason = "parse_error"
                        continue
                    parsed_file = True
                    stats["records_parsed"] += 1
                    stats["lenient_parses"] += int(lenient)
                    metrics = record.get("latentppi_scorer_pred_metrics")
                    if "latentppi_scorer_pred_metrics" not in record:
                        metric = record.get("latentppi_scorer_pred_metric_name")
                        metrics = ({metric: record.get("latentppi_scorer_pred_metric")}
                                   if isinstance(metric, str) and metric else {})
                    if not isinstance(metrics, dict):
                        stats["invalid_metric_objects"] += 1
                        metrics = {}
                    metric_names.update(metrics)
                    name = record.get("input_name")
                    if not isinstance(name, str) or not name:
                        stats["missing_identity_records"] += 1
                        file_reason = "missing_identity"
                        continue
                    name = canonical.get(name.casefold(), name)
                    if attempt and name != owner:
                        stats["foreign_attempt_records"] += 1
                        continue
                    seed = seed_key(record.get("seed"))
                    if incomplete(record):
                        stats["incomplete_dropped"] += 1
                        failed_systems[name] = ("incomplete", path, line_number, sidecar)
                        continue
                    row = {k: v for k, v in record.items() if k not in PRED_FIELDS}
                    row.update(derive_record(record))
                    if seed is None:
                        row["status"] = "missing_seed"
                    row.update(system_name=name, seed=seed, source_file=str(path), source_line=line_number)
                    row.update(sidecar)
                    if attempt.get("attempt_id"):
                        row.update(attempt_id=attempt["attempt_id"], run_id=attempt.get("run_id"),
                                   requested_recycles=attempt.get("num_recycles"))
                    for metric, value in metrics.items():
                        row[metric + "_pred"] = finite_number(value)
                    key = (name, seed)
                    records[key] = row
                    counts[key] += 1
        except OSError:
            stats["unreadable_files"] += 1
            file_reason = "read_error"
        stats["files_parsed"] += int(parsed_file)
        if owner:
            failed_systems.setdefault(owner, (file_reason, path, None, sidecar))
        elif not parsed_file:
            stats["unassigned_failure_files"] += 1
    stats["duplicate_keys"] = sum(count > 1 for count in counts.values())
    for name, attempt in attempts.items():
        if attempt.get("exit_code") == 0:
            for seed in attempt.get("completed_seeds", []):
                key = (name, seed_key(seed))
                if key not in records:
                    records[key] = dict(system_name=name, seed=key[1], status="missing_timing",
                        gated=False, diffusion_skipped=False, structure_confirmed=True,
                        attempt_id=attempt.get("attempt_id"), run_id=attempt.get("run_id"),
                        requested_recycles=attempt.get("num_recycles"))
    seen = {key[0] for key in records}
    for name in sorted((manifests.keys() | failed_systems.keys()) - seen):
        reason, path, line, env = failed_systems.get(name, ("not_run", None, None, {}))
        records[name, None] = dict(system_name=name, seed=None, status=reason, gated=False,
                                   source_file=str(path) if path else None, source_line=line, **env)
    for (name, _), row in records.items():
        if attempts.get(name, {}).get("attempt_id"):
            row["attempt_id"] = attempts[name]["attempt_id"]
        row["chain_a_length"] = row.get("latentppi_chain_len_a")
        row["chain_b_length"] = row.get("latentppi_chain_len_b")
        row.update(manifests.get(name, {}))
        if "total_length" not in row:
            a, b = (finite_number(row.get(k)) for k in ("chain_a_length", "chain_b_length"))
            row["total_length"] = a + b if a is not None and b is not None else None
    pred_columns = [metric + "_pred" for metric in sorted(metric_names)]
    extras = sorted({k for row in records.values() for k in row} - set(BASE_FIELDS) - set(pred_columns))
    frame = pd.DataFrame(list(records.values())).reindex(columns=[*BASE_FIELDS, *extras, *pred_columns])
    for column in pred_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Canonical integer columns remain nullable, including manifest-only rows.
    for column in ("line_number", "chain_a_length", "chain_b_length", "total_length", "source_line"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame = frame.sort_values(["system_name", "seed"], kind="stable", na_position="last").reset_index(drop=True)
    return frame, sorted(metric_names), stats


def join_ipsae(frame, metrics, path, rule, stats):
    # The parser uses this same lock while appending or extending the header.
    with Path(str(path) + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        summary = pd.read_csv(path, dtype={"system": str, "seed": str}, keep_default_na=False)
    if not {"system", "seed"}.issubset(summary.columns):
        raise ValueError(f"{path}: expected system and seed columns")
    metadata = {"system", "seed", "sample_num", "model_num", "rank", "recycle", "attempt_id",
                "ipsae_file", "af3_json", "af3_cif", "Chn1", "Chn2", "PAE", "Dist", "Type",
                "pae_cutoff", "dist_cutoff", "Model", "n0res", "n0chn", "n0dom", "d0res",
                "d0chn", "d0dom", "nres1", "nres2", "dist1", "dist2", "requested_recycles"}
    common = sorted(set(summary.columns) - metadata)
    stats["predicted_metrics_without_ground_truth"] = sorted(set(metrics) - set(common))
    summary["seed"] = summary["seed"].map(seed_key)
    unusable = summary["seed"].isna() | summary["system"].eq("")
    stats["ipsae_rows_missing_join_key"] = int(unusable.sum())
    summary = summary.loc[~unusable].copy()
    if "attempt_id" in frame:
        wanted = frame[["system_name", "seed", "attempt_id"]].copy()
        wanted["attempt_id"] = wanted.attempt_id.fillna("")
        if "attempt_id" not in summary:
            summary["attempt_id"] = ""
        summary = summary.merge(wanted.rename(columns={"system_name": "system"}).drop_duplicates(),
                                on=["system", "seed", "attempt_id"], how="inner")
    selected = select_ipsae_rows(summary, rule)
    selected = selected[["system", "seed", *common]].rename(
        columns={"system": "system_name", **{m: m + "_true" for m in common}})
    for metric in common:
        selected[metric + "_true"] = selected[metric + "_true"].map(finite_number)
    joined = frame.merge(selected, on=["system_name", "seed"], how="left",
                         validate="many_to_one", indicator="_ipsae_match")
    eligible = ~joined["gated"].fillna(False).astype(bool)
    stats["systems_matched_ipsae"] = joined.loc[
        joined["_ipsae_match"].eq("both") & eligible, "system_name"].nunique()
    joined.loc[~eligible, [m + "_true" for m in common]] = None
    return joined.drop(columns="_ipsae_match"), [m + "_true" for m in common]


def aggregate_pairs(frame, metric_columns, rank_by, agg):
    grouped = frame.groupby("system_name", sort=True, dropna=False)
    pairs = grouped.size().rename("n_records").to_frame()
    pairs["n_seeds_ok"] = frame["status"].eq("ok").groupby(frame["system_name"]).sum()
    pairs["gated"] = grouped["gated"].all()
    pairs["n_seeds_gated"] = grouped["gated"].sum()
    pairs["not_run"] = frame["status"].eq("not_run").groupby(frame["system_name"]).all()
    for field in ("line_number", "chain_a_length", "chain_b_length", "total_length"):
        pairs[field] = grouped[field].first()
    if metric_columns:
        pairs = pairs.join(grouped[metric_columns].agg(agg))
        pairs = pairs.join(grouped[metric_columns].max().add_suffix("_max"))
    pairs = pairs.reset_index()
    rank_column = rank_by + "_pred" if rank_by else None
    if rank_column:
        pairs["rank"] = pairs[rank_column].rank(method="min", ascending=False).astype("Int64")
        n_ranked = pairs[rank_column].notna().sum()
        pairs["rank_pct"] = pairs["rank"] / n_ranked if n_ranked else None
        pairs = pairs.sort_values([rank_column, "system_name"], ascending=[False, True],
                                  na_position="last", kind="stable")
    else:
        pairs["rank"] = pd.Series(pd.NA, index=pairs.index, dtype="Int64")
        pairs["rank_pct"] = None
    return pairs.reset_index(drop=True)


def write_csv(frame, path, compressed):
    """Replace only a completed output file; interrupted writes leave the old CSV intact."""
    serial = frame.copy()
    for column in serial:
        if serial[column].dtype == object:
            serial[column] = serial[column].map(
                lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list)) else value)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".metrics-", delete=False) as handle:
            temporary = Path(handle.name)
            if compressed:
                import gzip
                # No temporary filename or current timestamp embedded in gzip headers.
                with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as zipped:
                    serial.to_csv(zipped, index=False)
            else:
                serial.to_csv(handle, index=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--rank-by", help="Discovered metric name; inferred only if exactly one is present")
    parser.add_argument("--agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--jobs-tsv", type=Path)
    parser.add_argument("--lengths-csv", type=Path)
    parser.add_argument("--ipsae-summary", type=Path)
    parser.add_argument("--ipsae-row-select", default="max_ipsae",
                        help="max_ipsae (default), unique, or JSON where/order_by rule; see METRICS.md")
    parser.add_argument("--glob", default="**/prediction_timings.jsonl")
    parser.add_argument("--gzip", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.root.is_dir():
            raise ValueError(f"root is not a directory: {args.root}")
        def detected(explicit, filename):
            return explicit if explicit is not None else (
                args.root / filename if (args.root / filename).is_file() else None)
        jobs = detected(args.jobs_tsv, "jobs.tsv")
        lengths = detected(args.lengths_csv, "af3_lengths.csv")
        ipsae = detected(args.ipsae_summary, "ipsae_summary_af3.csv")
        frame, metrics, stats = collect(args.root, jobs, lengths, args.glob)
        rank_by = args.rank_by
        if rank_by is None and len(metrics) == 1:
            rank_by = metrics[0]
        if rank_by is not None and rank_by not in metrics:
            raise ValueError(f"rank-by metric {rank_by!r} absent; available metrics: {', '.join(metrics) or '(none)'}")
        if rank_by is None and len(metrics) > 1:
            raise ValueError(f"--rank-by is required for multiple metrics; available metrics: {', '.join(metrics)}")
        true_columns = []
        stats["predicted_metrics_without_ground_truth"] = metrics
        if ipsae:
            frame, true_columns = join_ipsae(frame, metrics, ipsae, args.ipsae_row_select, stats)
        metric_columns = sorted([m + "_pred" for m in metrics] + true_columns)
        frame = frame[[c for c in frame if c not in metric_columns] + metric_columns]
        pairs = aggregate_pairs(frame, metric_columns, rank_by, args.agg)
        stats.update({
            "pairs_written": len(pairs),
            "pairs_gated": int(pairs["gated"].sum()),
            "pairs_not_run": int(pairs["not_run"].sum()),
            "pairs_null_rank_by": int(pairs[rank_by + "_pred"].isna().sum()) if rank_by else len(pairs),
            "rows_inconsistent_recycle_timing": int(frame["recycle_timing_consistent"].eq(False).sum()),
            "rows_written": len(frame),
        })
        for key in ("files_found", "files_parsed", "records_parsed", "incomplete_dropped",
                    "lenient_parses", "duplicate_keys", "systems_matched_ipsae"):
            stats.setdefault(key, 0)
        out = args.out_dir or args.root
        out.mkdir(parents=True, exist_ok=True)
        suffix = ".csv.gz" if args.gzip else ".csv"
        write_csv(frame, out / ("metrics_all" + suffix), args.gzip)
        write_csv(pairs, out / ("metrics_pairs" + suffix), args.gzip)
        if not args.quiet:
            print(json.dumps(dict(stats), sort_keys=True), file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
