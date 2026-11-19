#!/usr/bin/env python3
"""Configure, submit, resume, inspect, and clean a protein screen."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate settings instead of silently replacing earlier YAML values."""


def unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in mapping:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate YAML key: {key}", key_node.start_mark)
            mapping[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(None, None, "YAML keys must be scalar", key_node.start_mark) from exc
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


@dataclass
class Inputs:
    conda_env: str = "colabfold"
    fasta_file: str = ""
    output_base: str = ""
    proteome_msa_dir: str = ""
    af3_num_recycles: int = 1
    af3_num_diffusion_samples: int = 1
    max_identity: float = 95.0
    filter_non_standard: bool = True
    force: bool = False
    recreate_manifest: bool = False
    recreate_commands: bool = False
    skip_if_msa_exists: bool = True


@dataclass
class Paths:
    script_dir: str = str(Path(__file__).resolve().parent)
    cmd_file: str = "af3_cmds.txt"
    af3_input_dir: str = "af3_inputs"
    af3_output_dir: str = "af3_predictions"
    summary_tsv: str = ""
    shard_dir: str = "h5_shards"
    ipsae_outroot: str = "ipsae_scores_af3"
    ipsae_summary: str = "ipsae_summary_af3.csv"


@dataclass
class AF3:
    image: str = "/path/to/af3_latentppi_sf_v303.sif"
    model_parameters_dir: str = "/path/to/af3/weights"
    databases_dir: str = "/path/to/af3/databases"


@dataclass
class Model:
    path: str


@dataclass
class Scorer:
    rank_by: str = ""
    threshold: float = 0.9
    models: dict[str, Model] = field(default_factory=dict)
    predict_metrics: list[str] = field(default_factory=list)
    manifest_path: str = ""


@dataclass
class Stages:
    save_embeddings: bool = False
    run_ipsae_parse: bool = True
    run_pool: bool = False
    run_compress: bool = True
    pool_seed: int = 1
    keep_npz: bool = False
    compress_embeddings: bool = False


@dataclass
class Cluster:
    job_name: str = ""  # Optional display prefix; empty preserves generated names.
    screen_cluster: str = ""
    inputs_cluster: str = ""
    partition: str = "your_gpu"
    cpu_only: bool = False
    gres: str = "gpu:1"
    constraint: str = ""
    gpu_tiers: list = field(default_factory=list)
    max_total_residues: int | None = None
    retry_failed_once: bool = False
    retry_partition: str = ""
    retry_gres: str = ""
    cpus_per_task: int = 4
    # Keep these internal/snapshot names stable for already-submitted jobs.
    # Public YAML names are normalized by cluster_settings() below.
    array_chunk: int = 4
    max_submit_jobs: int = 0
    submit_reserve: int = 2
    poll_interval: int = 60
    dispatch_mode: str = "chained"
    dispatcher_cpus_per_task: int = 1
    dispatcher_time_limit: str = "01:00:00"
    offset: int = 0
    max_array_size: int = 1001
    array_spec_limit: int = 1024
    time_limit: str = "24:00:00"
    inputs_partition: str = "your_cpu"
    mail_user: str = ""


@dataclass
class Output:
    rank_confident_iptm: bool = True
    metrics_dir: str = "metrics"
    refresh_seconds: int = 300
    known_interactors: str = ""
    write_summary: bool = True
    legacy_csv: bool = False


@dataclass
class Config:
    inputs: Inputs = field(default_factory=Inputs)
    paths: Paths = field(default_factory=Paths)
    af3: AF3 = field(default_factory=AF3)
    runner: str = "latent"
    scorer: Scorer = field(default_factory=Scorer)
    stages: Stages = field(default_factory=Stages)
    cluster: Cluster = field(default_factory=Cluster)
    output: Output = field(default_factory=Output)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pair_count: int | None = None
    proteome_count: int = 0
    model_paths: dict[str, str] = field(default_factory=dict)
    checkpoint_metadata: dict = field(default_factory=dict)
    rank_by: str = ""

    def print(self):
        for label, messages in (("INFO", self.notes), ("WARNING", self.warnings), ("ERROR", self.errors)):
            for message in messages:
                print(f"{label}: {message}", file=sys.stderr)
        print(f"Validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s)", file=sys.stderr)


def resolve_path(value, base):
    if not value:
        return ""
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return str((path if path.is_absolute() else base / path).resolve())


def cluster_settings(values, report):
    """Accept clearer YAML names without changing saved snapshots or dispatcher fields."""
    values = dict(values)  # Never mutate a caller's saved snapshot.
    aliases = {
        "screen_partition": "partition",
        "max_concurrent_tasks": "array_chunk",
        "user_queue_budget": "max_submit_jobs",
        "reserved_queue_slots": "submit_reserve",
    }
    for public, legacy in aliases.items():
        if public not in values:
            continue
        value = values.pop(public)
        if legacy in values:
            report.errors.append(f"cluster: specify only one of {public} and legacy {legacy}")
            continue
        if public == "user_queue_budget":
            # null is the public off switch; legacy zero stays valid for old YAML/JSON.
            if value is None:
                value = 0
            elif type(value) is not int or value < 1:
                report.errors.append("cluster.user_queue_budget: expected a positive integer "
                                     "or null to disable the budget check")
                continue
        values[legacy] = value
    for key in ("partition", "inputs_partition", "retry_partition"):
        if isinstance(values.get(key), str):
            values[key] = ",".join(part.strip() for part in values[key].split(","))
    if isinstance(values.get("gpu_tiers"), list):
        values["gpu_tiers"] = [dict(t) if isinstance(t, dict) else t for t in values["gpu_tiers"]]
    for tier in values.get("gpu_tiers", []) if isinstance(values.get("gpu_tiers", []), list) else []:
        if isinstance(tier, dict) and isinstance(tier.get("partition"), str):
            tier["partition"] = ",".join(p.strip() for p in tier["partition"].split(","))
    return values


def snapshot_config(raw):
    """Keep already-submitted snapshots on their original scheduling behavior."""
    cluster = dict(raw.get("cluster", {}))
    if "dispatch_mode" not in cluster:
        cluster["dispatch_mode"] = "persistent"
        cluster.setdefault("dispatcher_cpus_per_task", cluster.get("cpus_per_task", 4))
        cluster.setdefault("dispatcher_time_limit", "7-00:00:00")
    return {**raw, "cluster": cluster}


def load_config(path, report, raw=None):
    """Resolve into dataclasses while collecting schema errors across all sections."""
    config = Config()
    try:
        if raw is None:
            raw = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        report.errors.append(f"config: {exc}")
        raw = {}
    if not isinstance(raw, dict):
        report.errors.append("config must be a YAML mapping")
        raw = {}
    known = {f.name for f in fields(config)}
    for key in raw.keys() - known:
        report.errors.append(f"unknown config key: {key}")
    for section in fields(config):
        name = section.name
        if name == "runner":
            value = raw.get(name, config.runner)
            if value not in ("latent", "full"):
                report.errors.append("runner must be latent or full")
            else:
                config.runner = value
            continue
        obj = getattr(config, name)
        values = raw.get(name, {})
        if not isinstance(values, dict):
            report.errors.append(f"{name} must be a mapping")
            continue
        if name == "cluster":
            values = cluster_settings(values, report)
        allowed = {f.name for f in fields(obj)}
        for key in values.keys() - allowed:
            report.errors.append(f"unknown config key: {name}.{key}")
        for key, value in values.items():
            if key not in allowed or (name == "scorer" and key in ("models", "predict_metrics")):
                continue
            if name == "scorer" and key == "rank_by" and values.get("manifest_path"):
                continue
            default = getattr(obj, key)
            if name == "cluster" and key == "max_total_residues":
                if value is None or (type(value) is int and value > 0):
                    setattr(obj, key, value)
                else:
                    report.errors.append("cluster.max_total_residues must be a positive integer or null")
                continue
            valid = (type(value) is type(default) or
                     (type(default) is float and type(value) is int))
            if not valid or (isinstance(value, str) and key != "rank_by" and any(c in value for c in "\0\n\r")):
                report.errors.append(f"{name}.{key}: expected {type(default).__name__} without NUL/newlines")
            else:
                setattr(obj, key, value)
    base = path.resolve().parent
    for key in ("fasta_file", "output_base", "proteome_msa_dir"):
        value = getattr(config.inputs, key)
        if not value:
            report.errors.append(f"inputs.{key} is required")
        setattr(config.inputs, key, resolve_path(value, base))
    config.paths.script_dir = resolve_path(config.paths.script_dir, base)
    run_base = Path(config.inputs.output_base) if config.inputs.output_base else base
    for f in fields(config.paths):
        if f.name != "script_dir":
            setattr(config.paths, f.name, resolve_path(getattr(config.paths, f.name), run_base))
    for f in fields(config.af3):
        setattr(config.af3, f.name, resolve_path(getattr(config.af3, f.name), base))
    config.output.metrics_dir = resolve_path(config.output.metrics_dir, run_base)
    config.output.known_interactors = resolve_path(config.output.known_interactors, base)
    config.scorer.manifest_path = resolve_path(config.scorer.manifest_path, Path(config.paths.script_dir))
    scorer = raw.get("scorer", {})
    scorer = scorer if isinstance(scorer, dict) else {}
    if config.scorer.manifest_path:
        # External manifest is authoritative; YAML models/predict_metrics are unused.
        return config
    models = scorer.get("models", {})
    if not isinstance(models, dict):
        report.errors.append("scorer.models must be a mapping of metric names to {path: ...}")
        models = {}
    for metric, entry in models.items():
        if not isinstance(metric, str) or not metric:
            report.errors.append("scorer.models keys must be nonempty strings")
        elif (not isinstance(entry, dict) or set(entry) != {"path"}
              or not isinstance(entry["path"], str) or not entry["path"]):
            report.errors.append(f"scorer.models[{metric!r}] must contain a nonempty path")
        else:
            try:
                config.scorer.models[metric] = Model(resolve_path(entry["path"], Path(config.paths.script_dir)))
            except (OSError, ValueError) as exc:
                report.errors.append(f"scorer.models[{metric!r}].path: {exc}")
    requested = scorer.get("predict_metrics", sorted(config.scorer.models))
    if not isinstance(requested, list) or any(not isinstance(m, str) or not m for m in requested):
        report.errors.append("scorer.predict_metrics must be a list of nonempty metric names")
    else:
        config.scorer.predict_metrics = requested
        if len(set(requested)) != len(requested):
            report.errors.append("scorer.predict_metrics contains duplicates")
    return config


def readable_file(path, label, report):
    if not path or not Path(path).is_file() or not os.access(path, os.R_OK):
        report.errors.append(f"{label}: missing or unreadable file: {path or '(unset)'}")
        return False
    return True


def directory(path, label, report):
    if not path or not Path(path).is_dir() or not os.access(path, os.R_OK | os.X_OK):
        report.errors.append(f"{label}: missing or unreadable directory: {path or '(unset)'}")
        return False
    return True


def writable(path, label, report):
    if not path:
        report.errors.append(f"{label}: output path is empty")
        return False
    ancestor = Path(path)
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        report.errors.append(f"{label}: not writable or creatable: {path}")
        return False
    return True


def fasta_count(path):
    count, length = 0, 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if len(line) == 1 or (count and not length):
                raise ValueError("empty FASTA header or sequence")
            count += 1
            length = 0
        else:
            sequence = "".join(line.split())
            if not count or not re.fullmatch(r"[A-Za-z.*-]+", sequence):
                raise ValueError("invalid FASTA sequence or missing header")
            length += len(sequence)
    if not count or not length:
        raise ValueError("empty FASTA sequence")
    return count


def run_check(command, timeout=15):
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


CHECKPOINT_CODE = r'''
import collections, codecs, json, pickle, sys, types
try:
    import torch
    import numpy as np
except Exception as exc:
    print(json.dumps({"__runtime__": str(exc)}))
    sys.exit(0)
errors = {}
# Repository bundles store feature_mean/feature_std as NumPy arrays. Allow only
# these numerical reconstruction types, never arbitrary checkpoint globals.
reconstruct = np.core.multiarray._reconstruct
scalar = np.core.multiarray.scalar
allowed = [np.ndarray, np.dtype, reconstruct, scalar]
allowed += [type(np.dtype(t)) for t in (np.float32, np.float64, np.int32, np.int64, np.bool_)]
# NumPy 2 renamed core to _core; old bundles still use the former pickle names.
if reconstruct.__module__ != "numpy.core.multiarray":
    allowed += [(reconstruct, "numpy.core.multiarray._reconstruct"),
                (scalar, "numpy.core.multiarray.scalar")]
# Older installed PyTorch versions lack safe_globals. Use an explicitly
# restricted Unpickler there, not the default unrestricted pickle loader.
known = {("collections", "OrderedDict"): collections.OrderedDict,
         ("_codecs", "encode"): codecs.encode,
         ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
    known[module, "_reconstruct"] = reconstruct
    known[module, "scalar"] = scalar
for name in ("_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3", "_rebuild_parameter"):
    function = getattr(getattr(torch, "_utils", None), name, None)
    if function is not None:
        known["torch._utils", name] = function
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return known[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f"unsupported checkpoint global: {module}.{name}") from None
restricted = types.ModuleType("latentppi_restricted_pickle")
restricted.Unpickler = RestrictedUnpickler
restricted.load = lambda handle, **kwargs: RestrictedUnpickler(handle, **kwargs).load()
for metric, path in json.loads(sys.argv[1]).items():
    try:
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals(allowed):
                bundle = torch.load(path, map_location="cpu", weights_only=True)
        else:
            bundle = torch.load(path, map_location="cpu", weights_only=False, pickle_module=restricted)
        if len(sys.argv) > 2 and sys.argv[2] == "metadata":
            errors[metric] = {k: bundle.get(k) for k in
                              ("metric", "input_recycle", "target_recycle", "clip_bounds")}
    except Exception as exc:
        errors[metric] = str(exc)
print(json.dumps(errors))
'''


def load_checkpoints(model_paths, config, metadata=False):
    """Deserialize with existing PyTorch; never install a host dependency or unpickle unsafely."""
    if not model_paths:
        return {}
    if importlib.util.find_spec("torch") is not None:
        command = [sys.executable, "-c", CHECKPOINT_CODE, json.dumps(model_paths)]
    elif shutil.which("singularity") or shutil.which("apptainer"):
        runtime = shutil.which("singularity") or shutil.which("apptainer")
        parents = sorted({str(Path(p).parent) for p in model_paths.values()})
        command = [runtime, "exec"]
        for parent in parents:
            command += ["--bind", f"{parent}:{parent}:ro"]
        command += [config.af3.image, "uv", "run", "python3", "-c", CHECKPOINT_CODE, json.dumps(model_paths)]
    elif shutil.which("conda"):
        command = [shutil.which("conda"), "run", "--no-capture-output", "-n", config.inputs.conda_env,
                   "python", "-c", CHECKPOINT_CODE, json.dumps(model_paths)]
    else:
        return {"__runtime__": "checkpoint loading needs an existing PyTorch Python, AF3 container runtime, or configured conda environment"}
    if metadata:
        command.append("metadata")
    try:
        result = run_check(command, timeout=120)
        if result.returncode:
            return {"__runtime__": result.stderr.strip() or f"loader exited {result.returncode}"}
        result_data = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(result_data, dict):
            raise ValueError("loader did not return a mapping")
        return result_data
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {"__runtime__": str(exc)}


def validate_scorer(config, report):
    scorer = config.scorer
    if not math.isfinite(scorer.threshold) or not 0 <= scorer.threshold <= 1:
        report.errors.append("scorer.threshold must be within [0, 1]")
    if config.runner == "full":
        report.notes.append("runner=full: scorer models are unused; checkpoint checks skipped")
        return
    model_paths = {}
    if scorer.manifest_path:
        if readable_file(scorer.manifest_path, "scorer.manifest_path", report):
            try:
                data = json.loads(Path(scorer.manifest_path).read_text())
                models = data.get("models") if isinstance(data, dict) else None
                if not isinstance(models, dict) or not models:
                    raise ValueError("manifest models must be a nonempty mapping")
                rank_by = data.get("rank_by")
                if not isinstance(rank_by, str) or rank_by not in models:
                    raise ValueError(f"manifest rank_by must be a models key; available: {sorted(models)}")
                report.rank_by = rank_by
                parent = Path(scorer.manifest_path).parent
                for metric, entry in models.items():
                    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                        report.errors.append(f"manifest metric {metric!r}: expected a path")
                        continue
                    relative = Path(entry["path"])
                    target = (parent / relative).resolve()
                    components = [parent.joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1)]
                    if (relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(parent)
                            or any(p.is_symlink() for p in components)):
                        report.errors.append(f"manifest metric {metric!r}: path must stay relative to the bind-mounted manifest directory")
                        continue
                    model_paths[metric] = str(target)
            except (OSError, ValueError) as exc:
                report.errors.append(f"scorer.manifest_path: {exc}")
    else:
        available = sorted(scorer.models)
        if not scorer.predict_metrics:
            report.errors.append(f"scorer.predict_metrics must not be empty; available metrics: {available}")
        for metric in scorer.predict_metrics:
            if metric not in scorer.models:
                report.errors.append(f"unknown predicted metric {metric!r}; available metrics: {available}")
            else:
                model_paths[metric] = scorer.models[metric].path
        if scorer.rank_by not in scorer.predict_metrics:
            report.errors.append(f"rank_by {scorer.rank_by!r} is absent from predict_metrics; available metrics: {available}")
        report.rank_by = scorer.rank_by
    readable = {metric: path for metric, path in model_paths.items()
                if readable_file(path, f"checkpoint for metric {metric!r} (available: {sorted(model_paths)})", report)}
    for metric, info in load_checkpoints(readable, config, metadata=True).items():
        if isinstance(info, dict):
            report.checkpoint_metadata[metric] = info
        else:
            report.errors.append(f"checkpoint load {metric!r}: {info}; available metrics: {sorted(model_paths)}")
    report.model_paths = model_paths
    report.notes.append(f"Scorer rank_by={report.rank_by!r}; predicted metrics={sorted(model_paths)}")


def msa_references(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("unpairedMsaPath", "pairedMsaPath") and child:
                yield child
            else:
                yield from msa_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from msa_references(child)


def check_gpu_request(request, inventory, report, label):
    """Check configured per-node GPU inventory, not transient idle capacity."""
    wanted = []
    for part in request.split(","):
        if not part.startswith("gpu:"):
            continue
        match = re.fullmatch(r"gpu:(?:(?P<kind>[^:]+):)?(?P<count>\d+)", part)
        if not match:
            report.warnings.append(f"{label}: cannot verify GPU request {part!r}")
            continue
        wanted.append((match['kind'], int(match['count'])))
    if not wanted:
        return
    for node in inventory:
        counts = {}
        for match in re.finditer(r"(?:^|,)gpu:(?:([^:,()]+):)?(\d+)(?:\([^)]*\))?", node):
            kind, count = match.groups()
            counts[kind] = counts.get(kind, 0) + int(count)
        if all((counts.get(kind, 0) if kind else sum(counts.values())) >= count
               for kind, count in wanted):
            return
    report.warnings.append(f"{label}: GPU request {request!r} is not advertised on any eligible node; "
                           "check the GPU type/count and partition (this is not an idle-GPU check)")


def validate(config, report, *, inputs_only=False, force_inputs=False):
    inputs, paths, cluster = config.inputs, config.paths, config.cluster
    if config.output.refresh_seconds < 0:
        report.errors.append("output.refresh_seconds must be >= 0 (0 disables periodic updates)")
    if config.output.known_interactors:
        readable_file(config.output.known_interactors, "output.known_interactors", report)
    if cluster.job_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cluster.job_name):
        report.errors.append("cluster.job_name must start with a letter or digit and contain "
                             "only letters, digits, underscores, dots, or hyphens")
    for name in ("screen_cluster", "inputs_cluster"):
        value = getattr(cluster, name)
        if value and (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value) or value == "all"):
            report.errors.append(f"cluster.{name} must be empty or a single Slurm cluster name (not 'all')")
    for label, value in (("inputs.conda_env", inputs.conda_env),
                         ("paths.script_dir", paths.script_dir), ("cluster.screen_partition", cluster.partition),
                         ("cluster.inputs_partition", cluster.inputs_partition)):
        if not value:
            report.errors.append(f"{label} must not be empty")
    for label, value in (("screen_partition", cluster.partition),
                         ("inputs_partition", cluster.inputs_partition)):
        if any(not part or any(c.isspace() for c in part) for part in value.split(",")):
            report.errors.append(f"cluster.{label}: expected comma-separated partition names without empty entries")
    if cluster.cpu_only:
        if cluster.gres:
            report.notes.append("cluster.cpu_only: ignoring cluster.gres for CPU AF3 screening")
    elif not cluster.gres:
        report.errors.append("cluster.gres must not be empty unless cluster.cpu_only is true")
    from gpu_routing import tier_errors
    routing_errors = tier_errors(cluster.gpu_tiers)
    report.errors.extend(routing_errors)
    tiers = cluster.gpu_tiers if not routing_errors else []
    if cluster.gpu_tiers:
        if cluster.cpu_only:
            report.errors.append("cluster.gpu_tiers requires GPU screening")
        if cluster.retry_partition or cluster.retry_gres:
            report.errors.append("With gpu_tiers, omit retry_partition/retry_gres; retries use the final tier")
    if cluster.retry_failed_once:
        if cluster.cpu_only:
            report.errors.append("cluster.retry_failed_once requires GPU screening (cpu_only: false)")
        if not cluster.gpu_tiers and (not cluster.retry_partition or not cluster.retry_gres):
            report.errors.append("retry_failed_once requires cluster.retry_partition and cluster.retry_gres")
    if cluster.retry_partition and any(not part or any(c.isspace() for c in part)
                                       for part in cluster.retry_partition.split(",")):
        report.errors.append("cluster.retry_partition: expected comma-separated partition names without empty entries")
    if cluster.dispatch_mode not in ("chained", "persistent"):
        report.errors.append("cluster.dispatch_mode must be chained or persistent")
    if cluster.max_total_residues is not None and (
            type(cluster.max_total_residues) is not int or cluster.max_total_residues <= 0):
        report.errors.append("cluster.max_total_residues must be a positive integer or null")
    if cluster.dispatch_mode == "chained" and cluster.screen_cluster != cluster.inputs_cluster:
        report.errors.append("chained dispatch requires matching screen_cluster and inputs_cluster; "
                             "use dispatch_mode: persistent for separate clusters")
    for label, value, minimum in (
        ("inputs.af3_num_recycles", inputs.af3_num_recycles, 0),
        ("inputs.af3_num_diffusion_samples", inputs.af3_num_diffusion_samples, 1),
        ("stages.pool_seed", config.stages.pool_seed, 0),
        ("cluster.cpus_per_task", cluster.cpus_per_task, 1),
        ("cluster.dispatcher_cpus_per_task", cluster.dispatcher_cpus_per_task, 1),
        ("cluster.max_array_size", cluster.max_array_size, 2),
        ("cluster.array_spec_limit", cluster.array_spec_limit, 8),
        ("cluster.user_queue_budget (legacy cluster.max_submit_jobs)", cluster.max_submit_jobs, 0),
        ("cluster.reserved_queue_slots (legacy cluster.submit_reserve)", cluster.submit_reserve, 0),
        ("cluster.poll_interval", cluster.poll_interval, 1),
        ("cluster.max_concurrent_tasks (legacy cluster.array_chunk)", cluster.array_chunk, 1), ("cluster.offset", cluster.offset, 0)):
        if value < minimum:
            report.errors.append(f"{label} must be >= {minimum}")
    if cluster.max_submit_jobs > 0 and cluster.submit_reserve >= 0:
        same_cluster = cluster.screen_cluster == cluster.inputs_cluster
        uncertain_cluster = not same_cluster and not (cluster.screen_cluster and cluster.inputs_cluster)
        dispatcher_slots = int(same_cluster) + int(cluster.dispatch_mode == "chained")
        capacity = cluster.max_submit_jobs - cluster.submit_reserve - dispatcher_slots
        if capacity < 1:
            report.errors.append(
                "cluster.user_queue_budget (legacy cluster.max_submit_jobs) must allow at least one array task after subtracting "
                "cluster.reserved_queue_slots (legacy cluster.submit_reserve) and " + str(dispatcher_slots) + " dispatcher slot(s); "
                f"use at least {cluster.submit_reserve + dispatcher_slots + 1}. "
                "Chained mode also reserves a slot for its continuation.")
        elif capacity < cluster.array_chunk:
            report.warnings.append(
                f"cluster.max_concurrent_tasks (legacy cluster.array_chunk)={cluster.array_chunk} requests concurrent tasks, but "
                f"cluster.user_queue_budget (legacy cluster.max_submit_jobs) minus reserve and dispatcher leaves at most {capacity} "
                "array task slot(s), before unrelated jobs. Increase the budget or reduce concurrency.")
        if uncertain_cluster:
            report.warnings.append(
                "Only one of cluster.screen_cluster and cluster.inputs_cluster is explicit; "
                "if the default resolves to the same cluster, the dispatcher consumes one more "
                "budget slot. Allow room for it or specify both cluster names.")
    if cluster.max_array_size >= 2 and cluster.array_chunk > cluster.max_array_size - 1:
        report.warnings.append(
            f"cluster.max_concurrent_tasks (legacy cluster.array_chunk)={cluster.array_chunk} exceeds the {cluster.max_array_size - 1} "
            "tasks allowed by cluster.max_array_size; one active array cannot reach that concurrency.")
    if not re.fullmatch(r"(?:\d+-)?\d+:[0-5]\d:[0-5]\d", cluster.dispatcher_time_limit):
        report.errors.append("cluster.dispatcher_time_limit must use [days-]hours:minutes:seconds")
    if not math.isfinite(inputs.max_identity) or not 0 <= inputs.max_identity <= 100:
        report.errors.append("inputs.max_identity must be within [0, 100]")
    report.notes.append(f"max_identity={inputs.max_identity}: run_pipeline.py skips pairs above this threshold")
    if readable_file(inputs.fasta_file, "inputs.fasta_file", report):
        try:
            count = fasta_count(inputs.fasta_file)
            if count != 1:
                report.errors.append(f"inputs.fasta_file must contain exactly one sequence; found {count}")
        except (OSError, ValueError) as exc:
            report.errors.append(f"inputs.fasta_file: {exc}")
    if directory(inputs.proteome_msa_dir, "inputs.proteome_msa_dir", report):
        report.proteome_count = sum(p.is_file() for p in Path(inputs.proteome_msa_dir).glob("*.a3m"))
        if not report.proteome_count:
            report.errors.append("inputs.proteome_msa_dir contains no .a3m files")
    root = Path(inputs.output_base) if inputs.output_base else None
    if root and writable(str(root), "inputs.output_base", report):
        try:
            (root / "slurm_logs").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            report.errors.append(f"slurm_logs: {exc}")
        for msa in sorted(root.glob("*.a3m")):
            try:
                count = sum(line.startswith(">") for line in msa.read_text().splitlines())
                if count == 1:
                    report.warnings.append(f"SINGLE-SEQUENCE QUERY MSA: {msa}; target has no coevolution signal")
                elif count == 0:
                    report.errors.append(f"query A3M has no sequence headers: {msa}")
            except (OSError, ValueError) as exc:
                report.errors.append(f"query A3M {msa}: {exc}")
        try:
            from query_msa import discover
            discover(Path(inputs.fasta_file), root)
        except (OSError, ValueError) as exc:
            report.errors.append(f"query MSA: {exc}")
    for filename in ("run_pipeline.py", "screen.sh", "query_msa.py", "screen_workflow.py",
                     "latentppi.py", "collect_metrics.py", "screen_results.py", "record_run_env.py", "run_env_capture.sh",
                     "af3_pipeline_helpers.sh", "parse_af3_command.py",
                     "run_af3_latent_compress_pool.sh" if config.runner == "latent" else "run_af3_compress_pool.sh"):
        readable_file(str(Path(paths.script_dir) / filename), f"paths.script_dir/{filename}", report)
    if (inputs_only or force_inputs) and root:
        for key, expected in (("cmd_file", root / "af3_cmds.txt"),
                              ("af3_input_dir", root / "af3_inputs")):
            if Path(getattr(paths, key)) != expected:
                report.errors.append(f"stage 0 generates {expected}; cannot regenerate custom paths.{key}")
    if inputs_only:
        if not re.fullmatch(r"(?:\d+-)?\d+:[0-5]\d:[0-5]\d", cluster.time_limit):
            report.errors.append("cluster.time_limit must use [days-]hours:minutes:seconds")
        return report
    command_path = Path(paths.cmd_file) if paths.cmd_file else None
    if force_inputs:
        report.notes.append("Stage 0 will regenerate inputs and commands")
    elif command_path and command_path.is_file():
        try:
            lines = command_path.read_text().splitlines()
            report.pair_count = len(lines)
            if any(not line.strip() for line in lines):
                report.errors.append("cmd_file must have no blank task lines")
            if lines and cluster.offset >= len(lines):
                report.errors.append("cluster.offset must be less than the cmd_file line count")
        except OSError as exc:
            report.errors.append(f"cmd_file: {exc}")
    elif command_path and command_path.exists():
        report.errors.append(f"cmd_file is not a file: {command_path}")
    else:
        report.notes.append("Stage 0 pending: cmd_file does not exist; array size is not yet known")
        if root and command_path != root / "af3_cmds.txt":
            report.errors.append("missing custom cmd_file: stage 0 only generates output_base/af3_cmds.txt")
    report.notes.append(f"Proteome A3Ms: {report.proteome_count}; prepared command lines: "
                        f"{report.pair_count if report.pair_count is not None else 'pending stage 0'}")
    if report.pair_count is not None or Path(paths.af3_input_dir).exists():
        directory(paths.af3_input_dir, "paths.af3_input_dir", report)
    else:
        report.notes.append("Stage 0 pending: af3_input_dir will be generated")
        if root and Path(paths.af3_input_dir) != root / "af3_inputs":
            report.errors.append("missing custom af3_input_dir: stage 0 only generates output_base/af3_inputs")
    readable_file(config.af3.image, "af3.image", report)
    directory(config.af3.model_parameters_dir, "af3.model_parameters_dir", report)
    directory(config.af3.databases_dir, "af3.databases_dir", report)
    for key in ("af3_output_dir", "shard_dir", "ipsae_outroot"):
        writable(getattr(paths, key), f"paths.{key}", report)
    writable(str(Path(paths.ipsae_summary).parent), "paths.ipsae_summary parent", report)
    if Path(paths.ipsae_summary).exists() and (not Path(paths.ipsae_summary).is_file()
                                             or not os.access(paths.ipsae_summary, os.W_OK)):
        report.errors.append(f"paths.ipsae_summary is not a writable file: {paths.ipsae_summary}")
    writable(config.output.metrics_dir, "output.metrics_dir", report)
    if config.stages.run_pool and not config.stages.save_embeddings:
        report.errors.append("stages.run_pool requires stages.save_embeddings: true")
    if config.stages.run_pool:
        readable_file(paths.summary_tsv, "paths.summary_tsv (required for pooling)", report)
    for source in ([] if force_inputs else sorted(Path(paths.af3_input_dir).glob("*.json"))[:2]):
        try:
            data = json.loads(source.read_text())
            for reference in msa_references(data):
                if not isinstance(reference, str) or not Path(reference).is_absolute():
                    report.errors.append(f"MSA path in {source} must be absolute: {reference!r}")
                else:
                    readable_file(reference, f"MSA referenced by {source.name}", report)
        except (OSError, ValueError) as exc:
            report.errors.append(f"AF3 input {source}: {exc}")
    report.notes.append("MSA path checks run on the submit host; compute-node visibility is not proven")
    conda = shutil.which("conda")
    if conda:
        try:
            result = run_check([conda, "env", "list", "--json"])
            if result.returncode:
                raise ValueError(result.stderr.strip() or "conda env list failed")
            envs = json.loads(result.stdout).get("envs", [])
            if not any(Path(p).name == inputs.conda_env or p == inputs.conda_env for p in envs):
                report.errors.append(f"inputs.conda_env not found: {inputs.conda_env}; available: {envs}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            report.errors.append(f"conda environment check: {exc}")
    else:
        report.warnings.append("conda not on submit host PATH; confirm inputs.conda_env on the compute node")
    sinfo = shutil.which("sinfo")
    if sinfo:
        try:
            targets = {(cluster.inputs_cluster, cluster.inputs_partition)}
            if tiers:
                targets.update((cluster.screen_cluster, tier["partition"]) for tier in tiers)
            else:
                targets.add((cluster.screen_cluster, cluster.partition))
            if cluster.retry_partition:
                targets.add((cluster.screen_cluster, cluster.retry_partition))
            for target, partition in sorted(targets):
                command = [sinfo, "-h", "-N", "-o", "%P|%a|%G"]
                if target:
                    command.append("-M" + target)
                result = run_check(command)
                if result.returncode:
                    raise ValueError(f"cluster {target or 'default'}: " + (result.stderr.strip() or "sinfo failed"))
                rows = [line.strip().replace("*|", "|").split("|", 2)
                        for line in result.stdout.splitlines() if "|" in line]
                available = {row[0]: row[1] for row in rows}
                requests = []
                if target == cluster.screen_cluster:
                    if not tiers and partition == cluster.partition and not cluster.cpu_only:
                        requests.append(("gres", cluster.gres))
                    if partition == cluster.retry_partition and cluster.retry_gres:
                        requests.append(("retry_gres", cluster.retry_gres))
                    requests.extend((f"gpu_tiers[{i}].gres", tier["gres"])
                                    for i, tier in enumerate(tiers) if tier["partition"] == partition)
                for label, request in requests:
                    inventory = [row[2] for row in rows if len(row) == 3
                                 and row[0] in partition.split(",") and row[1] == "up"]
                    check_gpu_request(request, inventory, report,
                                      f"cluster.{label} on cluster {target or 'default'}, partitions {partition}")
                for name in partition.split(","):
                    if available.get(name) != "up":
                        report.errors.append(f"partition {name!r} is absent/inaccessible/down on cluster {target or 'default'}; sinfo: {available}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            report.errors.append(f"partition check: {exc}")
    else:
        report.warnings.append("sinfo not on submit host PATH; partition/GPU availability could not be checked")
    if not re.fullmatch(r"(?:\d+-)?\d+:[0-5]\d:[0-5]\d", cluster.time_limit):
        report.errors.append("cluster.time_limit must use [days-]hours:minutes:seconds")
    validate_scorer(config, report)
    bind_directories = [str(Path(paths.af3_input_dir).parent), paths.af3_output_dir,
                        config.af3.model_parameters_dir, config.af3.databases_dir,
                        inputs.proteome_msa_dir]
    if config.runner == "latent":
        bind_directories.append(str(Path(config.scorer.manifest_path).parent) if config.scorer.manifest_path
                                else str(Path(inputs.output_base) / "latentppi_config"))
    for path in sorted(set(bind_directories)):
        if "," in path or ":" in path:
            report.errors.append(f"Singularity --bind cannot represent a directory containing ',' or ':': {path}")
    return report


def environment(config, report, artifact_dir):
    i, p, a, s = config.inputs, config.paths, config.af3, config.stages
    env = dict(CONDA_ENV=i.conda_env, FASTA_FILE=i.fasta_file, OUTPUT_BASE=i.output_base,
               PROTEOME_MSA_DIR=i.proteome_msa_dir, AF3_SCRIPT_DIR=p.script_dir,
               AF3_NUM_RECYCLES=i.af3_num_recycles, AF3_NUM_DIFFUSION_SAMPLES=i.af3_num_diffusion_samples,
               MAX_IDENTITY=i.max_identity, FILTER_NON_STANDARD=i.filter_non_standard,
               FORCE=i.force, RECREATE_MANIFEST=i.recreate_manifest,
               RECREATE_COMMANDS=i.recreate_commands, SKIP_IF_MSA_EXISTS=i.skip_if_msa_exists,
               CMD_FILE=p.cmd_file, CMD_BASE_DIR=i.output_base, AF3_INPUT_DIR=p.af3_input_dir,
               AF3_MSA_DIR=i.proteome_msa_dir, AF3_OUTPUT_DIR=p.af3_output_dir,
               SUMMARY_TSV=p.summary_tsv, SHARD_DIR=p.shard_dir, IPSAE_OUTROOT=p.ipsae_outroot,
               IPSAE_SUMMARY=p.ipsae_summary, AF3_IMAGE=a.image,
               AF3_MODEL_PARAMETERS_DIR=a.model_parameters_dir, AF3_DATABASES_DIR=a.databases_dir,
               RUN_IPSAE_PARSE=s.run_ipsae_parse, RUN_POOL=s.run_pool, RUN_COMPRESS=s.run_compress,
               SAVE_EMBEDDINGS=s.save_embeddings,
               POOL_SEED=s.pool_seed, KEEP_NPZ=s.keep_npz, COMPRESS_EMBEDDINGS=s.compress_embeddings,
               AF3_CPU_ONLY=config.cluster.cpu_only,
               OFFSET=config.cluster.offset, DEFAULT_NUM_RECYCLES=i.af3_num_recycles, DRY_RUN=0,
               PYTHON_BIN=sys.executable, IPSAE_PYTHON=sys.executable, POOL_PYTHON=sys.executable,
               IPSAE_PARSER_PY=str(Path(p.script_dir) / "parse_ipsae.py"),
               IPSAE_PY=str(Path(p.script_dir) / "ipsae.py"), POOL_PY=str(Path(p.script_dir) / "pool_embeddings.py"),
               SINGULARITY_BIN="singularity", ZSTD_BIN="zstd", ZSTD_LEVEL=19, ZSTD_EMB_LEVEL=3,
               ZSTD_THREADS=config.cluster.cpus_per_task, IPSAE_PAE=15, IPSAE_DIST=15,
               IPSAE_FAILED_LOG=str(Path(p.ipsae_outroot) / "failed.txt"),
               IPSAE_DONE_LEDGER=str(Path(p.ipsae_outroot) / "done_ledger.txt"),
               IPSAE_FAIL_LEDGER=str(Path(p.ipsae_outroot) / "fail_ledger.txt"),
               LATENTPPI_RANK_BY=report.rank_by,
               PATH=os.environ.get("PATH", os.defpath))
    # Submission uses an isolated environment. Preserve runtime basics explicitly.
    for key in ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LD_LIBRARY_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    if not p.summary_tsv:
        env["SUMMARY_TSV"] = str(Path(i.output_base) / "dataset_summary.tsv")
    if config.runner == "latent" and report.model_paths:
        manifest = (Path(config.scorer.manifest_path) if config.scorer.manifest_path
                    else artifact_dir / "scorer_manifest.json")
        filename = (os.path.relpath(report.model_paths[report.rank_by], manifest.parent)
                    if config.scorer.manifest_path else checkpoint_name(report.rank_by))
        container_manifest = "/root/scorer/" + manifest.name
        env.update(AF3_SCORER_DIR=str(manifest.parent), AF3_SCORER_FILENAME=filename,
                   LATENTPPI_THRESHOLD=config.scorer.threshold, LATENTPPI_MODEL_MANIFEST=container_manifest,
                   SINGULARITYENV_LATENTPPI_MODEL_MANIFEST=container_manifest,
                   APPTAINERENV_LATENTPPI_MODEL_MANIFEST=container_manifest)
    return {key: str(int(value)) if isinstance(value, bool) else str(value)
            for key, value in sorted(env.items())}


def checkpoint_name(metric):
    return "checkpoints/" + hashlib.sha256(metric.encode()).hexdigest() + ".pt"


def write_artifact(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".render-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data.encode() if isinstance(data, str) else data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def submission_commands(config, report, artifact_dir):
    from screen_workflow import submit, prepare_scripts
    prepare_scripts(config, artifact_dir)
    env = environment(config, report, artifact_dir)
    common = dict(config=config, artifact_dir=artifact_dir, preview=True)
    return {
        "inputs": submit("inputs", Path(config.paths.script_dir) / "screen.sh", None, None,
                         env, [], **common),
        "dispatch": submit("dispatch", artifact_dir / "dispatch.sh", None, None,
                           {**env, "LATENTPPI_INPUTS_JOB_ID": "<inputs-job-id>"},
                           ["<inputs-job-id>"], **common),
    }


def render(config, report, source, *, artifact_dir=None, print_commands=True):
    artifact_dir = Path(artifact_dir) if artifact_dir else Path(config.inputs.output_base) / "latentppi_config"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.scorer.manifest_path or None
    if config.runner == "latent" and report.model_paths and not config.scorer.manifest_path:
        model_entries = {}
        for metric, source_path in sorted(report.model_paths.items()):
            relative = checkpoint_name(metric)
            destination = artifact_dir / relative
            destination.parent.mkdir(exist_ok=True)
            write_artifact(destination, Path(source_path).read_bytes())
            model_entries[metric] = {"path": relative}
        manifest_path = str(artifact_dir / "scorer_manifest.json")
        write_artifact(Path(manifest_path), json.dumps({"rank_by": report.rank_by, "models": model_entries},
                                                      indent=2, sort_keys=True) + "\n")
    env = environment(config, report, artifact_dir)
    write_artifact(artifact_dir / "env.sh", "# Generated by latentppi.py; source from Bash.\n" +
        "".join(f"export {key}={shlex.quote(value)}\n" for key, value in env.items()))
    write_artifact(artifact_dir / "sbatch.env", b"".join(
        (key + "=" + value).encode() + b"\0" for key, value in env.items()))
    write_artifact(artifact_dir / "run_options.json", json.dumps({"force": False}) + "\n")
    commands = submission_commands(config, report, artifact_dir)
    try:
        commit = run_check(["git", "-C", config.paths.script_dir, "rev-parse", "HEAD"])
        commit_hash = commit.stdout.strip() if commit.returncode == 0 else None
        dirty = run_check(["git", "-C", config.paths.script_dir, "status", "--porcelain", "--untracked-files=no"])
        git_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit_hash, git_dirty = None, None
    script_hashes = {}
    for filename in ("screen.sh", "run_af3_compress_pool.sh", "run_af3_latent_compress_pool.sh",
                     "af3_pipeline_helpers.sh", "run_pipeline.py", "latentppi.py", "screen_workflow.py",
                     "query_msa.py", "gpu_routing.py", "chained_dispatch.py", "record_run_env.py", "run_env_capture.sh", "screen_results.py", "screen_advance.py", "collect_metrics.py", "parse_ipsae.py"):
        path = Path(config.paths.script_dir) / filename
        if path.is_file():
            script_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    resolved = {"config": asdict(config), "source_config": str(source.resolve()),
                "git_commit": commit_hash, "git_dirty": git_dirty, "script_sha256": script_hashes,
                "environment": env, "scorer_manifest": manifest_path, "commands": commands,
                "effective_scorer": {"rank_by": report.rank_by, "model_paths": report.model_paths},
                "checkpoint_metadata": report.checkpoint_metadata,
                "validation": asdict(report),
                "checkpoint_sha256": {metric: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                      for metric, p in report.model_paths.items()}}
    write_artifact(artifact_dir / "resolved_config.json", json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    if print_commands:
        for stage, command in commands.items():
            print(f"# {stage}: would submit")
            print(command)
    print(f"Artifacts: {artifact_dir}", file=sys.stderr)
    return artifact_dir


def main(argv=None):
    from screen_workflow import launch, show_status, dispatch, screen_state, submission_lock
    import screen_cleanup
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    clean_parser = subparsers.add_parser("clean", aliases=["delete"],
                                         help="delete generated screen artifacts",
                                         description="Delete screen results; preserve prepared inputs unless --all is given.")
    screen_cleanup.add_arguments(clean_parser)
    for command in ("validate", "render", "inputs", "run", "advance", "status", "_dispatch", "_collect"):
        sub = subparsers.add_parser(command)
        sub.add_argument("config", type=Path)
        if command in ("inputs", "run"):
            sub.add_argument("--no-submit", action="store_true")
            sub.add_argument("--force-inputs", action="store_true")
        if command == "advance":
            sub.add_argument("--recycles", type=int, choices=range(1, 11), default=10)
            modes = sub.add_mutually_exclusive_group()
            modes.add_argument("--no-submit", action="store_true", help="list newly eligible pairs")
            modes.add_argument("--watch", action="store_true", help="advance new batches until the screen finishes")
        if command == "run":
            sub.add_argument("--force", action="store_true")
        if command == "status":
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument("--light", action="store_const", const="light",
                              dest="status_mode", default="light",
                              help="default: skip metrics recomputation and report file-based status")
            mode.add_argument("--heavy", action="store_const", const="heavy",
                              dest="status_mode",
                              help="recompute metrics files before reporting status")
    args = parser.parse_args(argv)
    if args.command in ("clean", "delete"):
        return screen_cleanup.run(args)
    report = Report()
    try:
        if args.command.startswith("_"):
            snapshot = args.config.resolve()
            resolved = json.loads((snapshot / "resolved_config.json").read_text())
            config = load_config(snapshot / "resolved_config.json", report, raw=snapshot_config(resolved["config"]))
            if report.errors:
                raise ValueError("; ".join(report.errors))
            if args.command == "_collect":
                screen_state(config, write_metrics=True, rank_by=resolved["effective_scorer"]["rank_by"], final=True)
            else:
                options = json.loads((snapshot / "run_options.json").read_text())
                config.scorer.rank_by = resolved["effective_scorer"]["rank_by"] or config.scorer.rank_by
                with submission_lock(config.inputs.output_base):
                    dispatch(config, snapshot, {**resolved["environment"], "LATENTPPI_RUN_SNAPSHOT": str(snapshot),
                                               "LATENTPPI_DISPATCH_STEP": os.environ.get("LATENTPPI_DISPATCH_STEP", "0")}, force=options["force"],
                             inputs_job=os.environ.get("LATENTPPI_INPUTS_JOB_ID"))
            return 0
        config = load_config(args.config, report)
        if report.errors:
            report.print()
            return 1
        force_inputs = getattr(args, "force_inputs", False)
        if force_inputs:
            config.inputs.force = config.inputs.recreate_manifest = config.inputs.recreate_commands = True
        if args.command != "status":
            validate(config, report, inputs_only=args.command == "inputs", force_inputs=force_inputs)
        report.print()
        if report.errors:
            return 1
        if args.command == "advance":
            from screen_advance import advance
            advance(config, report, args.recycles, args.no_submit, args.watch)
        elif args.command == "render":
            render(config, report, args.config)
        elif args.command == "status":
            show_status(config, heavy=args.status_mode == "heavy")
        elif args.command in ("run", "inputs"):
            launch(config, report, args.config, render, inputs_only=args.command == "inputs",
                   force_inputs=force_inputs, force=getattr(args, "force", False), preview=args.no_submit)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
