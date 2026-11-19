"""Screen orchestration. Only submit() speaks the sbatch submission protocol."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


class SubmitQuotaExceeded(RuntimeError):
    """A definite scheduler rejection that can safely be retried."""


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data.encode() if isinstance(data, str) else data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def submit(stage, script, cmd_file, array_range, env, depends_on, *, config,
           artifact_dir, preview=False, dependency="afterok", script_args=(), record=True, begin=None):
    """Submit one SLURM job and return its ID; preview returns the quoted command.

    This is the only boundary constructing sbatch arguments, encoding dependency
    semantics, or parsing submission responses. Callers specify a success or
    completion dependency policy, independent of the command-line representation.
    """
    if stage not in {"inputs", "dispatch", "array", "collector"}:
        raise ValueError(f"unknown stage: {stage}")
    if dependency not in {"afterok", "afterany"}:
        raise ValueError(f"unsupported dependency policy: {dependency}")
    if not preview and any(not re.fullmatch(r"\d+", job) for job in depends_on):
        raise ValueError("invalid dependency job ID")
    root, cluster = Path(config.inputs.output_base), config.cluster
    target_cluster = cluster.screen_cluster if stage == "array" else cluster.inputs_cluster
    env = {k: str(v) for k, v in env.items()}
    if stage == "array":
        env["AF3_CPU_ONLY"] = str(int(cluster.cpu_only))
        if cluster.cpu_only:
            # Bash ignores the runner's GPU SBATCH defaults. All allocation
            # options come from this submission, including the CPU partition.
            wrapper = Path(artifact_dir) / "cpu_array.sh"
            atomic_write(wrapper, "#!/bin/bash\nexec /bin/bash " +
                         shlex.quote(str(Path(script).resolve())) + ' "$@"\n')
            script = wrapper
    if any("\0" in k + v or "=" in k for k, v in env.items()):
        raise ValueError("environment names/values cannot contain NUL (or '=' in names)")
    digest = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]
    export = Path(artifact_dir) / "exports" / f"{stage}-{digest}.env"
    atomic_write(export, b"".join((k + "=" + v).encode() + b"\0" for k, v in sorted(env.items())))
    screen_id = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    # Names are display labels; tracking and dependencies use recorded Slurm IDs.
    job_prefix = cluster.job_name or f"latentppi-{screen_id}"
    args = ["sbatch", "--parsable", f"--job-name={job_prefix}-{stage}",
            "--export=ALL", "--chdir=" + str(root), "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={cluster.dispatcher_cpus_per_task if stage == 'dispatch' else cluster.cpus_per_task}",
            "--time=" + (cluster.dispatcher_time_limit if stage == "dispatch" else cluster.time_limit),
            "--partition=" + (cluster.partition if stage == "array" else cluster.inputs_partition),
            "--output=" + str(root / f"slurm_logs/{stage}_%A_%a.out"),
            "--error=" + str(root / f"slurm_logs/{stage}_%A_%a.err")]
    if begin:
        args.append("--begin=" + begin)
    if target_cluster:
        args.append("--clusters=" + target_cluster)
    if stage == "array" and cluster.retry_failed_once:
        args.append("--no-requeue")
    if stage == "dispatch":
        # A scheduler restart would replay this snapshot while its last array
        # may still run. Resume through `run` after recorded jobs finish.
        args.append("--no-requeue")
    args += (["--mail-user=" + cluster.mail_user, "--mail-type=END,FAIL"]
             if cluster.mail_user else ["--mail-type=NONE"])
    if stage == "array":
        if not array_range:
            raise ValueError("array submission requires a nonempty range")
        if not cluster.cpu_only:
            args.append("--gres=" + cluster.gres)
            if cluster.constraint:
                args.append("--constraint=" + cluster.constraint)
        args.append("--array=" + array_range)
    if depends_on:
        args += ["--dependency=" + dependency + ":" + ":".join(depends_on), "--kill-on-invalid-dep=yes"]
    args.append(str(script))
    if cmd_file is not None:
        args.append(str(cmd_file))
    args.extend(map(str, script_args))
    if preview:
        command = ["env", "-i", *[k + "=" + v for k, v in sorted(env.items())], *args]
        return "(cd " + shlex.quote(str(root)) + " && " + shlex.join(command) + ")"
    # Pass values through the process environment, avoiding --export's comma parsing
    # and --export-file, which some cluster submission clients do not support.
    # Do not retry an ambiguous submission: a lost response may still have created a job.
    result = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        if any(code in result.stderr for code in (
                "QOSMaxSubmitJobPerUserLimit", "QOSMaxSubmitJobPerAccountLimit",
                "AssocMaxSubmitJobLimit", "QOSGrpSubmitJobsLimit")):
            # Only explicit rejections are retryable; never retry a lost response.
            raise SubmitQuotaExceeded(result.stderr.strip())
        raise RuntimeError(f"{stage} submission failed: {result.stderr.strip() or result.stdout.strip()}")
    match = re.fullmatch(r"(\d+)(?:;([\w.-]+))?", result.stdout.strip())
    if not match:
        raise RuntimeError(f"{stage}: unrecognized submission response {result.stdout!r}; inspect the queue before retrying")
    job_id = match.group(1)
    if record:
        entry = dict(job_id=job_id, stage=stage, submitted_at=time.time(), array_range=array_range,
                     cluster=match.group(2) or target_cluster,
                     offset=int(env.get("OFFSET", 0)), depends_on=depends_on, dependency=dependency,
                     export_file=str(export), args=args)
        if env.get("LATENTPPI_GPU_TIER"):
            entry.update(gpu_tier=env["LATENTPPI_GPU_TIER"],
                         gpu_route_reason=env.get("LATENTPPI_GPU_ROUTE_REASON"))
        with (Path(artifact_dir) / "submissions.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    print(f"{stage}: submitted job {job_id}")
    return job_id


def compact_indices(indices):
    ranges = []
    first = last = None
    for index in sorted(set(indices)):
        if last is not None and index == last + 1:
            last = index
            continue
        if last is not None:
            ranges.append(str(first) if first == last else f"{first}-{last}")
        first = last = index
    if last is not None:
        ranges.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(ranges)


def chunk_indices(indices, max_array_size=1001, max_spec_length=1024, concurrency=4, offset=0):
    """Return (OFFSET, local array spec); MaxArraySize is an exclusive index bound."""
    if max_array_size < 2 or max_spec_length < 3 or concurrency < 1 or offset < 0:
        raise ValueError("invalid array chunk limits")
    indices = sorted(set(indices))
    if any(type(index) is not int or index < 1 for index in indices):
        raise ValueError("manifest indices must be positive integers")
    capacity = max_array_size - 1
    batches, pending, current_offset = [], [], None
    suffix = f"%{concurrency}"
    for index in indices:
        if index <= offset:
            continue
        window = offset + ((index - offset - 1) // capacity) * capacity
        local = index - window
        if current_offset != window or len(compact_indices(pending + [local]) + suffix) > max_spec_length:
            if pending:
                batches.append((current_offset, compact_indices(pending) + suffix))
            current_offset, pending = window, []
        pending.append(local)
        if len(compact_indices(pending) + suffix) > max_spec_length:
            raise ValueError("array_spec_limit is too small to encode one task")
    if pending:
        batches.append((current_offset, compact_indices(pending) + suffix))
    return batches


def array_limit(config):
    """Use the configured cap, reduced by discoverable scheduler limits."""
    limit = config.cluster.max_array_size
    if shutil.which("scontrol"):
        try:
            command = ["scontrol"]
            if config.cluster.screen_cluster:
                command.append("--clusters=" + config.cluster.screen_cluster)
            result = subprocess.run(command + ["show", "config"], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                match = re.search(r"\bMaxArraySize\s*=\s*(\d+)", result.stdout)
                if match:
                    limit = min(limit, int(match.group(1)))
                tasks = re.search(r"\bmax_array_tasks=(\d+)", result.stdout)
                if tasks:
                    limit = min(limit, int(tasks.group(1)) + 1)
            else:
                print("WARNING: cannot read scheduler limits; using configured max_array_size", file=sys.stderr)
        except (OSError, subprocess.TimeoutExpired):
            print("WARNING: cannot read scheduler limits; using configured max_array_size", file=sys.stderr)
    return limit


def read_pairs(config, require_commands=False):
    root = Path(config.inputs.output_base)
    lengths = root / "af3_lengths.csv"
    if not lengths.is_file():
        raise ValueError("af3_lengths.csv is required; run inputs or use run --force-inputs")
    with lengths.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"system_name", "line_number"}.issubset(reader.fieldnames or []):
            raise ValueError("af3_lengths.csv needs system_name and line_number")
        pairs = list(reader)
    seen_names, seen_indices = set(), set()
    for pair in pairs:
        pair["line_number"] = int(pair["line_number"])
        if (not pair["system_name"] or pair["system_name"] in seen_names
                or pair["line_number"] < 1 or pair["line_number"] in seen_indices):
            raise ValueError("af3_lengths.csv contains empty/duplicate names or invalid task indices")
        seen_names.add(pair["system_name"])
        seen_indices.add(pair["line_number"])
    if require_commands:
        lines = Path(config.paths.cmd_file).read_text().splitlines()
        if any(not line.strip() for line in lines) or seen_indices != set(range(1, len(lines) + 1)):
            raise ValueError("af3_lengths.csv indices do not match the command file; regenerate inputs")
    return sorted(pairs, key=lambda p: p["line_number"])


def length_exclusions(config, pairs):
    """Return excluded command indices and lengths, without changing the manifest."""
    maximum = config.cluster.max_total_residues
    if maximum is None:
        return {}
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("cluster.max_total_residues must be a positive integer or null")
    excluded = {}
    for pair in pairs:
        raw = pair.get("total_length", "")
        if not re.fullmatch(r"[0-9]+", str(raw)) or int(raw) <= 0:
            raise ValueError("cluster.max_total_residues requires a positive integer total_length "
                             f"in af3_lengths.csv for {pair['system_name']}; regenerate inputs")
        length = int(raw)
        if length > maximum:
            excluded[pair["line_number"]] = length
    return excluded


def report_length_exclusions(config, excluded):
    if config.cluster.max_total_residues is not None:
        print(f"Size cutoff: {len(excluded):,} command(s) excluded above "
              f"{config.cluster.max_total_residues:,} total residues")


def screen_state(config, *, write_metrics=False, rank_by=None, final=False):
    if write_metrics:
        from screen_results import publication_lock
        with publication_lock(config.output.metrics_dir):
            return _screen_state(config, write_metrics=True, rank_by=rank_by, final=final)
    return _screen_state(config, rank_by=rank_by, final=final)


def _screen_state(config, *, write_metrics=False, rank_by=None, final=False):
    """Classify manifest completion independently of timing instrumentation quality."""
    import collect_metrics as collector
    pairs = read_pairs(config)
    excluded = length_exclusions(config, pairs)
    root, output = Path(config.inputs.output_base), Path(config.paths.af3_output_dir)
    jobs = root / "jobs.tsv"
    attempts = collector.load_attempts(output)
    frame, metrics, stats = collector.collect(output, jobs if jobs.is_file() else None,
                                              root / "af3_lengths.csv", attempts=attempts)
    latest = collector.load_attempts(output)
    changed = {name for name in attempts.keys() | latest.keys() if attempts.get(name) != latest.get(name)}
    if changed:
        # A task can restart while collection reads its append-only timing file.
        # Defer its scores until the next refresh instead of assigning them to
        # the wrong attempt. Other pairs remain available in this snapshot.
        import pandas as pd
        frame = frame.loc[~frame.system_name.isin(changed)].copy()
        placeholders = [dict(system_name=name, seed=None, status="incomplete", gated=False,
                             attempt_id=latest.get(name, {}).get("attempt_id")) for name in changed]
        frame = pd.concat([frame, pd.DataFrame(placeholders)], ignore_index=True)
        attempts = latest
    grouped = {name: group for name, group in frame.groupby("system_name", sort=False)}
    folders = {}
    if output.is_dir():
        for folder in output.iterdir():
            if folder.is_dir() and not folder.name.startswith("."):
                folders.setdefault(folder.name.casefold(), []).append(folder)
    counts = dict(total=len(pairs), completed=0, failed=0, gated=0, never_run=0, skipped_length=0)
    statuses, pending, never, timing_warnings = {}, [], [], {}
    for pair in pairs:
        name = pair["system_name"]
        rows = grouped.get(name)
        attempt = attempts.get(name, {})
        ok = rows[rows.status.eq("ok")] if rows is not None else None
        if rows is not None:
            warnings = rows[rows.status.isin(["missing_timing", "invalid_timing"])]
            if not warnings.empty:
                timing_warnings[name] = [
                    {"seed": row.seed, "status": row.status}
                    for row in warnings.itertuples()
                ]
                # A successful process and a terminal result record establish
                # completion independently of optional timing instrumentation.
                # An exit code alone must not promote a partial record.
                if attempt.get("exit_code") == 0:
                    def terminal_result(row):
                        if row.get("gated"):
                            metric = row.get("latentppi_scorer_rank_by")
                            if not isinstance(metric, str):
                                metric = rank_by or config.scorer.rank_by
                            return (isinstance(metric, str)
                                    and collector.finite_number(row.get(metric + "_pred")) is not None)
                        return row.get("diffusion_skipped") is False

                    accepted = warnings.apply(terminal_result, axis=1)
                    ok = rows[rows.status.eq("ok") | rows.index.isin(warnings[accepted].index)]
        expected = None
        if pair.get("json_path"):
            source = Path(pair["json_path"])
            source = source if source.is_absolute() else root / source
            try:
                seeds = json.loads(source.read_text()).get("modelSeeds")
                if isinstance(seeds, list) and seeds:
                    expected = {str(seed) for seed in seeds}
            except (OSError, ValueError, AttributeError):
                pass
        complete = ok is not None and not ok.empty
        if complete and expected is not None:
            complete = expected.issubset(set(ok.seed))
        if attempt and attempt.get("exit_code") != 0:
            complete = False
        if complete:
            counts["completed"] += 1
            gated = bool(ok.gated.all())
            counts["gated"] += int(gated)
            statuses[name] = "gated" if gated else "completed"
        elif pair["line_number"] in excluded and not (attempt and attempt.get("exit_code") is None):
            counts["skipped_length"] += 1
            statuses[name] = "skipped_length"
        else:
            attempted = (bool(attempt) or bool(folders.get(name.casefold()))
                         or (rows is not None and rows.status.ne("not_run").any()))
            status = "failed" if attempted else "never_run"
            counts[status] += 1
            statuses[name] = status
            pending.append(pair["line_number"])
            if not attempted:
                never.append(pair["line_number"])
    result = {**counts, "never_run_indices": never, "outstanding_indices": pending,
              "systems": statuses, "timing_warnings": timing_warnings, "collector": dict(stats)}
    if write_metrics:
        from screen_results import build_master, publish
        planned = {p["system_name"] for p in pairs}
        result["excluded_unplanned_records"] = int((~frame.system_name.isin(planned)).sum())
        frame = frame.loc[frame.system_name.isin(planned)].copy()
        columns = []
        summary = Path(config.paths.ipsae_summary)
        if summary.is_file():
            frame, columns = collector.join_ipsae(frame, metrics, summary, "max_ipsae", stats)
        ranking = rank_by or config.scorer.rank_by
        if ranking not in metrics:
            ranking = metrics[0] if len(metrics) == 1 and not ranking else None
        metric_columns = sorted([metric + "_pred" for metric in metrics] + columns)
        frame = frame[[c for c in frame if c not in metric_columns] + metric_columns]
        legacy = None
        if config.output.legacy_csv:
            legacy = (frame, collector.aggregate_pairs(frame, metric_columns, ranking, "mean"))
        result["collector"] = dict(stats)
        final_metric = rank_by or config.scorer.rank_by or "ipTM_af"
        result["rank_by"] = "ipTM_af" if config.output.rank_confident_iptm else final_metric
        master = build_master(config, pairs, frame, attempts, result, final_metric, final=final)
        result["results"] = publish(config, master, result, final_metric, legacy)
    return result


def active_jobs(config):
    """Query only jobs recorded for this screen; no scheduler access is needed for disk counts."""
    ids = {}
    for ledger in (Path(config.inputs.output_base) / "latentppi_runs").glob("*/submissions.jsonl"):
        for line in ledger.read_text().splitlines():
            try:
                entry = json.loads(line)
                job = entry["job_id"]
                if re.fullmatch(r"\d+", job):
                    ids.setdefault(entry.get("cluster", ""), set()).add(job)
            except (ValueError, KeyError, TypeError):
                continue
    if not ids:
        return [], None
    if not shutil.which("squeue"):
        return [], "squeue unavailable; current job state is unknown"
    jobs = []
    try:
        for target, cluster_ids in ids.items():
            ordered = sorted(cluster_ids, key=int)
            for start in range(0, len(ordered), 100):
                command = ["squeue", "--noheader", "--jobs=" + ",".join(ordered[start:start + 100]),
                           "--format=%i|%T|%j"]
                if target:
                    command.append("--clusters=" + target)
                result = subprocess.run(command, capture_output=True, text=True, timeout=15)
                if result.returncode and "Invalid job id" not in result.stderr:
                    return jobs, f"squeue failed on cluster {target or 'default'}: " + result.stderr.strip()
                for line in result.stdout.splitlines():
                    parts = line.strip().split("|", 2)
                    if len(parts) == 3:
                        jobs.append(dict(job_id=parts[0], state=parts[1], name=parts[2], cluster=target))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return jobs, f"squeue failed: {exc}"
    return jobs, None


def show_status(config, *, heavy=False):
    rank_by = config.scorer.rank_by
    if config.runner == "latent" and config.scorer.manifest_path:
        rank_by = json.loads(Path(config.scorer.manifest_path).read_text())["rank_by"]
    state = screen_state(config, write_metrics=heavy, rank_by=rank_by)
    jobs, warning = active_jobs(config)
    planned = state["total"]
    resolved = state["completed"]
    rejected = state["gated"]
    folded = max(resolved - rejected, 0)
    unresolved = state["failed"]
    never_run = state["never_run"]

    def pct(part, whole):
        return 0.0 if whole <= 0 else (100.0 * part / whole)

    if heavy:
        print("INFO: status is running in heavy mode; metrics were recomputed and written to disk.",
              file=sys.stderr)
    else:
        print("WARNING: status is running in light mode (default); metrics were not recomputed. "
              "Use --heavy to refresh all_results.tsv, ranked_survivors.tsv, and screen_status.json.",
              file=sys.stderr)

    print(f"Unique planned systems: {planned:,}")
    print(f"Folded (saved structure): {folded:,} ({pct(folded, planned):.2f}% of planned)")
    print(f"Rejected by scorer (diffusion skipped): {rejected:,} ({pct(rejected, planned):.2f}% of planned)")
    print(f"Unresolved output (running, failed, or incomplete): {unresolved:,} ({pct(unresolved, planned):.2f}% of planned)")
    print(f"No output directory yet: {never_run:,} ({pct(never_run, planned):.2f}% of planned)")
    print(f"Skipped for length: {state['skipped_length']:,}; "
          f"max_total_residues={config.cluster.max_total_residues}")
    print(f"Resolved outcomes: {resolved:,}; folded {pct(folded, resolved):.2f}%; rejected {pct(rejected, resolved):.2f}%")
    if state["timing_warnings"]:
        if heavy:
            print(f"WARNING: timing data incomplete or invalid for {len(state['timing_warnings'])} "
                  "system(s); see timing_warnings in screen_status.json and outcomes in all_results.tsv")
        else:
            print(f"WARNING: timing data incomplete or invalid for {len(state['timing_warnings'])} "
                  "system(s); run status --heavy to refresh timing_warnings in screen_status.json "
                  "and outcomes in all_results.tsv")
    print("never-run array indices: " + (compact_indices(state["never_run_indices"]) or "none"))
    print("outstanding array indices: " + (compact_indices(state["outstanding_indices"]) or "none"))
    for job in jobs:
        print(f"job {job['job_id']}: {job['state']} {job['name']}")
    if not jobs and not warning:
        print("queued/running jobs: none")
    if warning:
        print("WARNING: " + warning, file=sys.stderr)
    return state


@contextmanager
def submission_lock(root, *, filename=".latentppi_submit.lock", shared=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / filename).open("a") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if filename == ".latentppi_watch.lock":
                raise RuntimeError("advance watcher or cleanup is active for this screen") from exc
            raise RuntimeError("another submission for this screen is in progress") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_scripts(config, snapshot):
    snapshot = Path(snapshot)
    for stage, internal in (("dispatch", "_dispatch"), ("collector", "_collect")):
        command = [sys.executable, str(Path(config.paths.script_dir) / "latentppi.py"), internal, str(snapshot)]
        atomic_write(snapshot / (stage + ".sh"), "#!/bin/bash\nset -Eeuo pipefail\nexec " + shlex.join(command) + "\n")


def user_queue(config):
    """One row per task, including compressed pending arrays and unrelated jobs.

    Count the entire user's target-cluster queue conservatively rather than
    guessing the site's association/QOS hierarchy. A failed query is never empty.
    """
    command = ["squeue", "--local", "--noheader", "--array", "--all", "--states=all",
               "--user=" + str(os.getuid()), "--format=%F"]
    if config.cluster.screen_cluster:
        command.append("--clusters=" + config.cluster.screen_cluster)
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Cannot read user queue: " + result.stderr.strip())
    jobs = result.stdout.split()
    if any(not re.fullmatch(r"\d+", job) for job in jobs):
        raise RuntimeError("Unrecognized user queue response; stopping submission")
    return jobs


def wait_for_array(config, job):
    last_refresh = time.monotonic()
    while job in user_queue(config):
        interval = config.output.refresh_seconds
        if interval and time.monotonic() - last_refresh >= interval:
            refresh_results(config)
            last_refresh = time.monotonic()
        time.sleep(config.cluster.poll_interval)


def refresh_results(config):
    try:
        ranking = config.scorer.rank_by
        if config.runner == "latent" and config.scorer.manifest_path and not ranking:
            ranking = json.loads(Path(config.scorer.manifest_path).read_text())["rank_by"]
        screen_state(config, write_metrics=True, rank_by=ranking)
    except (OSError, ValueError, KeyError) as exc:
        print(f"WARNING: results refresh failed; keeping the last published files: {exc}", file=sys.stderr)


def dispatch(config, snapshot, env, *, force=False, inputs_job=None, preview=False):
    if config.cluster.dispatch_mode == "chained":
        from chained_dispatch import dispatch as dispatch_once
        return dispatch_once(config, snapshot, env, force=force, preview=preview)
    pairs = read_pairs(config, require_commands=True)
    excluded = length_exclusions(config, pairs)
    report_length_exclusions(config, excluded)
    state = screen_state(config)
    outstanding = [pair["line_number"] for pair in pairs] if force else state["outstanding_indices"]
    print(f"Pairs: {len(pairs)}; skipped as complete: {0 if force else state['completed']}; "
          f"outstanding: {len(outstanding)}; max_identity={config.inputs.max_identity}")
    retry_path = Path(config.inputs.output_base) / "latentppi_gpu_retries.json"
    retry_records = json.loads(retry_path.read_text()) if retry_path.exists() else {}
    names = {pair["line_number"]: pair["system_name"] for pair in pairs}
    eligible = [index for index in outstanding if index > config.cluster.offset and index not in excluded]
    remaining = eligible
    if config.cluster.retry_failed_once:
        remaining = [index for index in eligible if names[index] not in retry_records
                     and (force or state["systems"].get(names[index]) != "failed")]
    from gpu_routing import tier_errors, tier_index, resource_config
    cluster = config.cluster
    errors = tier_errors(cluster.gpu_tiers)
    if errors:
        raise ValueError("; ".join(errors))
    eligible_set = set(eligible)
    routes = {pair["line_number"]: tier_index(cluster.gpu_tiers, pair)
              for pair in pairs if pair["line_number"] in eligible_set} if cluster.gpu_tiers else {}
    limit = array_limit(config)
    runner = Path(config.paths.script_dir) / (
        "run_af3_latent_compress_pool.sh" if config.runner == "latent" else "run_af3_compress_pool.sh")
    arrays = []
    active = None
    if not preview and config.output.refresh_seconds:
        refresh_results(config)
    try:
        for retry in ([False, True] if cluster.retry_failed_once else [False]):
            submit_config = config
            if retry:
                submit_config = (resource_config(config, cluster.gpu_tiers[-1]) if cluster.gpu_tiers else
                                 replace(config, cluster=replace(cluster, partition=cluster.retry_partition,
                                                                 gres=cluster.retry_gres)))
                if preview:
                    print("# Failed pairs will be retried once with partition=" + submit_config.cluster.partition
                          + " gres=" + submit_config.cluster.gres)
                    continue
                after = screen_state(config)
                remaining = [index for index in eligible if names[index] not in retry_records
                             and after["systems"].get(names[index]) not in ("completed", "gated")]
            while remaining:
                capacity = len(remaining)
                if cluster.max_submit_jobs:
                    used = 0 if preview else len(user_queue(config))
                    capacity = cluster.max_submit_jobs - cluster.submit_reserve - used
                    if capacity <= 0:
                        print("Waiting for submission quota...", flush=True)
                        time.sleep(cluster.poll_interval)
                        continue
                tier = None
                if cluster.gpu_tiers:
                    route = len(cluster.gpu_tiers) - 1 if retry else routes[remaining[0]]
                    tier = cluster.gpu_tiers[route]
                    submit_config = resource_config(config, tier)
                    candidates = [i for i in remaining if retry or routes[i] == route][:capacity]
                else:
                    candidates = remaining[:capacity]
                route_env = ({"LATENTPPI_GPU_TIER": tier["name"],
                              "LATENTPPI_GPU_ROUTE_REASON": "retry" if retry else "total_length"}
                             if tier else {})
                offset, spec = chunk_indices(candidates, limit, cluster.array_spec_limit,
                                             cluster.array_chunk, cluster.offset)[0]
                count = sum(int(bounds[-1]) - int(bounds[0]) + 1
                            for bounds in (part.split("-") for part in spec.split("%")[0].split(",")))
                selected = candidates[:count]
                if retry and not preview:
                    # Reserve before sbatch: an ambiguous response must never cause a duplicate retry.
                    for index in selected:
                        retry_records[names[index]] = dict(state="reserved", reserved_at=time.time(),
                                                           partition=submit_config.cluster.partition, gres=submit_config.cluster.gres)
                    atomic_write(retry_path, json.dumps(retry_records, sort_keys=True) + "\n")
                try:
                    job = submit("array", runner, Path(config.paths.cmd_file), spec,
                                 {**env, **route_env, "OFFSET": str(offset), "LATENTPPI_GPU_RETRY": str(int(retry))}, [],
                                 config=submit_config, artifact_dir=snapshot, preview=preview)
                except SubmitQuotaExceeded:
                    if retry and not preview:
                        for index in selected:
                            retry_records.pop(names[index], None)
                        atomic_write(retry_path, json.dumps(retry_records, sort_keys=True) + "\n")
                    print("Submission quota changed; waiting before retrying.", flush=True)
                    time.sleep(cluster.poll_interval)
                    continue
                if preview:
                    print(job)
                    print("# Wait for this array to leave the queue before submitting the next.")
                    job = f"<array-{len(arrays) + 1}>"
                arrays.append(job)
                active = job
                if retry and not preview:
                    for index in selected:
                        retry_records[names[index]].update(state="submitted", job_id=job)
                    atomic_write(retry_path, json.dumps(retry_records, sort_keys=True) + "\n")
                # The chunk is a prefix of candidates, potentially split by index or
                # expression-length limits. Remove exactly its accepted task count.
                selected_set = set(selected)
                remaining = [i for i in remaining if i not in selected_set]
                if not preview:
                    wait_for_array(config, job)
                    if config.output.refresh_seconds:
                        refresh_results(config)
                active = None
    finally:
        # Completed IDs can age out of Slurm while later batches run. Depend only
        # on an array still being monitored when dispatch failed.
        if arrays or not remaining:
            job = submit("collector", Path(snapshot) / "collector.sh", None, None, env,
                         [active] if active else [],
                         config=config, artifact_dir=snapshot, preview=preview, dependency="afterany")
            if preview:
                print(job)
    return arrays


def check_chained_handoffs(root):
    """A lost sbatch response may hide an active job from the normal job ledger."""
    for path in (Path(root) / "latentppi_runs").glob("*/dispatch_state.json"):
        state = json.loads(path.read_text())
        pending = state.get("pending")
        if pending and not pending.get("job_id"):
            raise RuntimeError(f"Unresolved {pending['kind']} submission in {path}; "
                               "inspect/cancel uncertain jobs before clearing this pending marker and resuming")


def launch(config, report, source, render_fn, *, inputs_only=False, force_inputs=False,
           force=False, preview=False):
    root = Path(config.inputs.output_base)
    with submission_lock(root):
        jobs, warning = active_jobs(config)
        if jobs and not preview:
            raise RuntimeError("this screen still has active jobs; inspect status before resubmitting")
        if warning and not preview:
            raise RuntimeError(warning + "; cannot safely rule out duplicate submissions")
        if not preview and not inputs_only:
            check_chained_handoffs(root)
            from screen_results import ensure_compatible_screen
            ensure_compatible_screen(config, report)
        prepared = Path(config.paths.cmd_file).is_file() and Path(config.paths.af3_input_dir).is_dir()
        needs_inputs = inputs_only or force_inputs or not prepared
        if not needs_inputs:
            print("Inputs already exist; skipping stage 0 (use --force-inputs to regenerate).")
        snapshot = root / "latentppi_runs" / (("preview-" if preview else "run-") + uuid.uuid4().hex[:12])
        render_fn(config, report, source, artifact_dir=snapshot, print_commands=False)
        prepare_scripts(config, snapshot)
        env = json.loads((snapshot / "resolved_config.json").read_text())["environment"]
        env["LATENTPPI_RUN_SNAPSHOT"] = str(snapshot)
        atomic_write(snapshot / "run_options.json", json.dumps(dict(force=force)) + "\n")
        print(f"Run snapshot: {snapshot}")
        if needs_inputs:
            inputs_id = submit("inputs", Path(config.paths.script_dir) / "screen.sh", None, None, env, [],
                               config=config, artifact_dir=snapshot, preview=preview)
            if preview:
                print(inputs_id)
                inputs_id = "<inputs-job-id>"
            if not inputs_only:
                job = submit("dispatch", snapshot / "dispatch.sh", None, None,
                             {**env, "LATENTPPI_INPUTS_JOB_ID": inputs_id}, [inputs_id],
                             config=config, artifact_dir=snapshot, preview=preview)
                if preview:
                    print(job)
                    print("# Dispatcher reads the finished manifest, resumes outstanding tasks, and submits")
                    print("# arrays after input success, then a collector after every array terminates (success or failure).")
        else:
            job = submit("dispatch", snapshot / "dispatch.sh", None, None, env, [],
                         config=config, artifact_dir=snapshot, preview=preview)
            if preview:
                print(job)
                print("# Dispatcher submits one array and a dependent continuation, then exits."
                      if config.cluster.dispatch_mode == "chained" else
                      "# Persistent dispatcher waits for quota and completion.")
    return snapshot
