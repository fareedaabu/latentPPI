"""Record task hardware and publish metadata beside the actual AF3 output folder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import re

from screen_workflow import atomic_write


def structure_signature(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def fresh_structure(path, attempt):
    """Compare with pre-run files; avoid relying on wall-clock timestamp precision."""
    if "structure_baselines" in attempt:
        return attempt["structure_baselines"].get(str(Path(path).resolve())) != structure_signature(path)
    return Path(path).stat().st_mtime >= attempt["started_at"]


def record(mode, output, folder_name, json_path, exit_code=None, stage=None):
    output = Path(output)
    # The input JSON gives the unsplit, case-preserving system identifier.
    name = json.loads(Path(json_path).read_text())["name"]
    path = output / ".latentppi_attempts" / (hashlib.sha256(name.encode()).hexdigest() + ".json")
    if mode == "start":
        # Keep old attempts for inspection; the small top-level file is the latest one.
        if path.is_file():
            old = json.loads(path.read_text())
            old_id = old.get("attempt_id") or "legacy-" + uuid.uuid4().hex
            atomic_write(path.parent / "history" / path.stem / (old_id + ".json"),
                         json.dumps(old, sort_keys=True) + "\n")
        baselines, structures = {}, {}
        if output.is_dir():
            for folder in output.iterdir():
                if folder.is_dir() and folder.name.casefold() == folder_name.casefold():
                    for cif in folder.rglob("*.cif*"):
                        if cif.is_file():
                            structures[str(cif.resolve())] = structure_signature(cif)
                    for timing in folder.rglob("prediction_timings.jsonl"):
                        raw = timing.read_bytes()
                        baselines[str(timing.resolve())] = dict(size=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest(), lines=len(raw.splitlines()),
                            signature=structure_signature(timing))
        gpu = None
        command = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible and visible != "NoDevFiles":
            command += ["--id=" + visible]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu = ", ".join(dict.fromkeys(line.strip() for line in result.stdout.splitlines() if line.strip())) or None
        except (OSError, subprocess.TimeoutExpired):
            pass
        threshold = os.environ.get("LATENTPPI_THRESHOLD")
        data = dict(system_name=name, gpu_type=gpu,
                    gpu_retry=os.environ.get("LATENTPPI_GPU_RETRY") == "1", node=os.environ.get("SLURMD_NODENAME"),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                    rank_by=os.environ.get("LATENTPPI_RANK_BY") or None,
                    threshold=float(threshold) if threshold else None,
                    started_at=time.time(), exit_code=None, attempt_id=uuid.uuid4().hex,
                    run_id=Path(os.environ.get("LATENTPPI_RUN_SNAPSHOT", "")).name,
                    timing_baselines=baselines, structure_baselines=structures,
                    num_recycles=os.environ.get("LATENTPPI_REQUESTED_RECYCLES"),
                    input_sha256=hashlib.sha256(Path(json_path).read_bytes()).hexdigest())
        snapshot = os.environ.get("LATENTPPI_RUN_SNAPSHOT")
        if snapshot:
            resolved = json.loads((Path(snapshot) / "resolved_config.json").read_text())
            data["checkpoint_metadata"] = resolved.get("checkpoint_metadata", {})
    else:
        if not path.is_file():
            return
        data = json.loads(path.read_text())
        data.update(exit_code=int(exit_code), finished_at=time.time())
        data["stage"] = stage
        stderr = path.parent / (data.get("attempt_id", "unknown") + ".stderr")
        message = stderr.read_text(errors="replace") if stderr.is_file() else ""
        data["failure_reason"] = ("oom" if int(exit_code) and re.search(
            r"out of memory|RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY|oom-kill", message, re.I)
            else "other" if int(exit_code) else "")
        # Fresh sample structures establish completion even if AF3 omitted timings.
        completed = set()
        for folder in output.iterdir():
            if folder.is_dir() and folder.name.casefold() == folder_name.casefold():
                for cif in folder.glob("seed-*_sample-*/model.cif*"):
                    match = re.fullmatch(r"seed-(\d+)_sample-(\d+)", cif.parent.name)
                    if match and fresh_structure(cif, data):
                        completed.add(match[1])
        data["completed_seeds"] = sorted(completed)
    atomic_write(path, json.dumps(data, sort_keys=True) + "\n")
    folders = [p for p in output.iterdir() if p.is_dir() and p.name.casefold() == folder_name.casefold()]
    if len(folders) == 1:
        atomic_write(folders[0] / "run_env.json", json.dumps(data, sort_keys=True) + "\n")
    return data.get("attempt_id", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["start", "finish"])
    parser.add_argument("output")
    parser.add_argument("folder_name")
    parser.add_argument("json_path")
    parser.add_argument("exit_code", nargs="?")
    parser.add_argument("stage", nargs="?")
    args = parser.parse_args()
    attempt_id = record(args.mode, args.output, args.folder_name, args.json_path, args.exit_code, args.stage)
    if args.mode == "start":
        print(attempt_id)
