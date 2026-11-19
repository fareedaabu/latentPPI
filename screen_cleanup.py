"""Implementation of the latentppi clean/delete commands."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import latentppi
import screen_workflow
from screen_results import PUBLIC_FILES, publication_lock
import re


def contains(parent, child):
    return child == parent or parent in child.parents


def lexical_path(value, base):
    """Expand configuration paths without hiding symlinks through resolve()."""
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def reject_symlinks(path):
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"refusing cleanup through symlink: {component}")


def result_link(path, metrics):
    """Only our exact relative result aliases are allowed."""
    if path.parent != metrics or not path.is_symlink():
        return False
    reject_symlinks(metrics)
    reject_symlinks(metrics / ".generations")
    target = os.readlink(path)
    if path.name == "current":
        return re.fullmatch(r"\.generations/[0-9a-f]{32}", target) is not None
    return path.name in PUBLIC_FILES and target == "current/" + path.name and result_link(metrics / "current", metrics)


def load_config(source):
    raw = latentppi.yaml.load(source.read_text(), Loader=latentppi.UniqueKeyLoader)
    report = latentppi.Report()
    config = latentppi.load_config(source, report, raw=raw)
    if report.errors:
        raise ValueError("; ".join(report.errors))
    # The shared loader resolves symlinks. Check the original output spellings too.
    base = source.resolve().parent
    root = lexical_path(raw["inputs"]["output_base"], base)
    reject_symlinks(root)
    for key in ("cmd_file", "af3_input_dir", "af3_output_dir", "shard_dir",
                "ipsae_outroot", "ipsae_summary"):
        value = raw.get("paths", {}).get(key, getattr(latentppi.Paths(), key))
        if not value:
            raise ValueError(f"paths.{key} must not be empty for cleanup")
        reject_symlinks(lexical_path(value, root))
    metrics = raw.get("output", {}).get("metrics_dir", latentppi.Output().metrics_dir)
    if not metrics:
        raise ValueError("output.metrics_dir must not be empty for cleanup")
    reject_symlinks(lexical_path(metrics, root))
    return config


def build_plan(config, source, *, all_inputs=False, remove_query_msa=False):
    """Return existing, non-overlapping deletion targets; never delete the root."""
    root = Path(config.inputs.output_base)
    reject_symlinks(root)
    repo = Path(__file__).resolve().parent
    script_dir = Path(config.paths.script_dir)
    if root in (Path(root.anchor), Path.home().resolve(), repo, script_dir):
        raise ValueError(f"refusing cleanup of an unsafe screen root: {root}")
    if root.exists() and not root.is_dir():
        raise ValueError(f"screen root is not a directory: {root}")
    if remove_query_msa and not all_inputs:
        raise ValueError("--remove-query-msa requires --all")

    # Protect resources even when someone has placed them inside the screen.
    protected_files = [source.resolve(), Path(config.inputs.fasta_file)]
    protected_files += [Path(model.path) for model in config.scorer.models.values()]
    protected_dirs = [Path(config.inputs.proteome_msa_dir),
                      Path(config.af3.model_parameters_dir), Path(config.af3.databases_dir)]
    protected_files.append(Path(config.af3.image))
    if config.paths.summary_tsv:
        protected_files.append(Path(config.paths.summary_tsv))
    protected_files.append(root / "dataset_summary.tsv")
    if config.scorer.manifest_path:
        protected_dirs.append(Path(config.scorer.manifest_path).parent)

    query = root / (Path(config.inputs.fasta_file).stem + ".a3m")
    digest = hashlib.sha256(str(Path(config.inputs.fasta_file).resolve()).encode()).hexdigest()[:12]
    fallback_query = root / f"{Path(config.inputs.fasta_file).stem}_{digest}_query.a3m"
    if not remove_query_msa:
        protected_files += [query, fallback_query]
    if not all_inputs:
        protected_dirs.append(Path(config.paths.af3_input_dir))
        protected_files += [Path(config.paths.cmd_file), root / "jobs.tsv", root / "af3_lengths.csv"]

    directories = {Path(getattr(config.paths, key)) for key in
                   ("af3_output_dir", "shard_dir", "ipsae_outroot")}
    directories.update(root / name for name in
                       ("slurm_logs", "colab_predictions", "ipsae_scores", "latentppi_config", "latentppi_runs"))
    directories.update(root.glob(".query-msa.*"))
    candidates = list(directories)
    candidates += [Path(config.paths.ipsae_summary), Path(config.paths.ipsae_summary + ".lock")]
    candidates += [Path(config.output.metrics_dir) / name for name in
                   ("metrics_all.csv", "metrics_pairs.csv", "screen_status.json")]
    metrics = Path(config.output.metrics_dir)
    generations = metrics / ".generations"
    directories.add(generations)
    candidates += [generations, metrics / "current"] + [metrics / name for name in PUBLIC_FILES]
    candidates += [root / name for name in (
        "af3_job_durations.txt", "af3_failed_jobs.txt", "latentppi_gpu_retries.json",
        "colab_job_durations.txt", "colab_failed_jobs.txt", "job_submission_colab.txt",
        "job_submission_af3.txt",
        "done_ledger.txt", "done_ledger.txt.lock", "fail_ledger.txt", "fail_ledger.txt.lock",
        "failed.txt")]
    if all_inputs:
        directories.add(Path(config.paths.af3_input_dir))
        candidates += [Path(config.paths.af3_input_dir), Path(config.paths.cmd_file),
                       root / "jobs.tsv", root / "af3_lengths.csv", root / "screen_candidates.tsv",
                       root / "screen_identity.json"]
    if remove_query_msa:
        candidates += [query, fallback_query]

    plan = []
    for path in sorted(set(candidates)):
        safe_result_link = result_link(path, metrics)
        if not safe_result_link:
            reject_symlinks(path)
        if path == root or not contains(root, path):
            raise ValueError(f"cleanup target must be strictly inside the screen root: {path}")
        for directory in protected_dirs:
            if contains(directory, path) or contains(path, directory):
                raise ValueError(f"cleanup target {path} overlaps protected directory {directory}")
        for protected in [*protected_files, repo, script_dir, root / ".latentppi_submit.lock"]:
            if contains(path, protected):
                raise ValueError(f"cleanup target {path} contains protected path {protected}")
        if path.exists() or safe_result_link:
            if path in directories:
                if not path.is_dir():
                    raise ValueError(f"expected generated directory, refusing cleanup: {path}")
            elif not path.is_file() and not safe_result_link:
                raise ValueError(f"expected generated file, refusing cleanup: {path}")
            plan.append(path)

    # Inspect nested symlinks as well: never follow them or conceal shared inputs.
    for path in plan:
        if path.is_dir() and not path.is_symlink():
            def walk_error(error):
                raise error

            for directory, dirs, files in os.walk(path, followlinks=False, onerror=walk_error):
                for name in dirs + files:
                    child = Path(directory) / name
                    if child.is_symlink():
                        raise ValueError(f"refusing cleanup of directory containing symlink: {child}")
    plan = [p for p in plan if not any(other != p and contains(other, p) for other in plan)]
    # Preserve submission history if an earlier deletion fails.
    return sorted(plan, key=lambda p: (contains(p, root / "latentppi_runs"), str(p)))


def require_idle(config):
    root = Path(config.inputs.output_base)
    # active_jobs tolerates damaged ledger entries for status; deletion must not.
    for ledger in (root / "latentppi_runs").glob("*/submissions.jsonl"):
        for line in ledger.read_text().splitlines():
            try:
                job = json.loads(line)["job_id"]
                if not isinstance(job, str) or not job.isascii() or not job.isdigit():
                    raise ValueError("invalid job ID")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"cannot verify jobs from submission ledger {ledger}: {exc}") from exc
    jobs, warning = screen_workflow.active_jobs(config)
    if warning:
        raise RuntimeError(f"cannot establish that the screen is idle: {warning}")
    if jobs:
        ids = ", ".join(job["job_id"] for job in jobs)
        raise RuntimeError(f"screen has active Slurm jobs: {ids}; wait for completion or cancel them before cleanup")


@contextmanager
def cleanup_locks(config):
    """Keep parent, advance and batch submissions out until deletion finishes."""
    root = Path(config.inputs.output_base)
    with ExitStack() as locks:
        def lock(directory, filename=".latentppi_submit.lock"):
            reject_symlinks(directory / filename)
            locks.enter_context(screen_workflow.submission_lock(directory, filename=filename))

        lock(root)
        lock(root, ".latentppi_watch.lock")
        screens = [config]
        advance = root / "advance"
        if advance.is_dir():
            lock(advance)
            for batch in sorted(advance.glob("batch-*")):
                if batch.is_dir():
                    lock(batch)
                    screens.append(replace(config, inputs=replace(config.inputs, output_base=str(batch))))
        locks.enter_context(publication_lock(config.output.metrics_dir))
        yield screens


def require_all_idle(screens):
    for screen in screens:
        screen_workflow.check_chained_handoffs(screen.inputs.output_base)
        require_idle(screen)


def show_plan(root, plan):
    print(f"Screen: {root}")
    for path in plan:
        print(f"  DELETE {path}{'/' if path.is_dir() else ''}")
    if not plan:
        print("Nothing to remove.")


def add_arguments(parser):
    parser.add_argument("config", type=Path, help="screen YAML configuration")
    parser.add_argument("--dry-run", action="store_true", help="list deletions without changing files or querying Slurm")
    parser.add_argument("--all", dest="all_inputs", action="store_true", help="also remove prepared inputs, commands, and manifests")
    parser.add_argument("--remove-query-msa", action="store_true", help="also remove the cached query MSA (requires --all)")
    parser.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt; retain all safety checks")


def run(args):
    try:
        config = load_config(args.config)
        root = Path(config.inputs.output_base)
        options = dict(all_inputs=args.all_inputs, remove_query_msa=args.remove_query_msa)
        plan = build_plan(config, args.config, **options)
        if args.dry_run or not plan:
            show_plan(root, plan)
            if args.dry_run:
                print("Preview only. Actual cleanup also checks recorded Slurm jobs before deleting.")
            return 0
        with cleanup_locks(config) as screens:
            require_all_idle(screens)
            plan = build_plan(config, args.config, **options)
            show_plan(root, plan)
            if not args.yes:
                try:
                    answer = input("Permanently delete the listed artifacts? [y/N] ")
                except EOFError:
                    answer = ""
                if answer.strip().lower() not in ("y", "yes"):
                    print("Aborted.")
                    return 0
            # Recheck after the user has reviewed the plan. Keep the same lock inode.
            require_all_idle(screens)
            if build_plan(config, args.config, **options) != plan:
                raise RuntimeError("cleanup targets changed during confirmation; run cleanup again")
            for path in plan:
                try:
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                except OSError as exc:
                    raise RuntimeError(f"failed to remove {path}: {exc}; cleanup may be partial") from exc
            print("Cleanup complete. Configuration and source resources preserved.")
        return 0
    except (OSError, ValueError, RuntimeError, latentppi.yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
