"""Exercise real deletions only inside temporary screens; never submit jobs."""
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import screen_cleanup as clean
import latentppi
import screen_workflow as workflow
import yaml


class CleanScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="clean-screen-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "screen with spaces"
        self.source = self.base / "config.yaml"
        self.raw = dict(inputs=dict(fasta_file="target.fasta", output_base=self.root.name,
                                   proteome_msa_dir="proteome"), runner="full")
        (self.base / "target.fasta").write_text(">target\nAAAA\n")
        (self.base / "proteome").mkdir()
        self.write_config()
        for name in ("af3_predictions/pair/result.json", "af3_predictions/.latentppi_attempts/pair.json",
                     "h5_shards/recycle_1/embeddings_pair.h5", "ipsae_scores_af3/done_ledger.txt",
                     "ipsae_summary_af3.csv", "ipsae_summary_af3.csv.lock", "slurm_logs/job.out",
                     "af3_job_durations.txt", "af3_failed_jobs.txt", "latentppi_config/env.sh",
                     "latentppi_runs/run-old/resolved_config.json", ".query-msa.123/target.a3m",
                     "metrics/metrics_all.csv", "metrics/metrics_pairs.csv", "metrics/screen_status.json",
                     "af3_inputs/pair.json", "af3_cmds.txt", "af3_lengths.csv", "jobs.tsv",
                     "target.a3m", "dataset_summary.tsv", "notes.h5", "my_ledger.txt",
                     "metrics/notes.txt"):
            self.put(name)

    def put(self, name, content="fixture"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_config(self):
        self.source.write_text(yaml.safe_dump(self.raw))

    def run_clean(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = latentppi.main(["clean", str(self.source), *args])
        return code, output.getvalue()

    def test_default_removes_results_preserves_inputs_sources_and_unrelated_files(self):
        code, output = self.run_clean("--yes")
        self.assertEqual(code, 0, output)
        for name in ("af3_predictions", "h5_shards", "latentppi_runs", "latentppi_config",
                     "ipsae_scores_af3", "slurm_logs", "metrics/metrics_all.csv", ".query-msa.123"):
            self.assertFalse((self.root / name).exists(), name)
        for name in ("af3_inputs/pair.json", "af3_cmds.txt", "jobs.tsv", "af3_lengths.csv",
                     "target.a3m", "dataset_summary.tsv", "notes.h5", "my_ledger.txt", "metrics/notes.txt",
                     ".latentppi_submit.lock"):
            self.assertTrue((self.root / name).is_file(), name)
        self.assertTrue(self.source.is_file())
        self.assertTrue((self.base / "target.fasta").is_file())
        self.assertTrue((self.base / "proteome").is_dir())
        self.assertEqual(self.run_clean("--yes")[0], 0)

    def test_all_and_optional_query_removal(self):
        code, output = self.run_clean("--all", "--yes")
        self.assertEqual(code, 0, output)
        for name in ("af3_inputs", "af3_cmds.txt", "jobs.tsv", "af3_lengths.csv"):
            self.assertFalse((self.root / name).exists())
        self.assertTrue((self.root / "target.a3m").is_file())
        code, output = self.run_clean("--all", "--remove-query-msa", "--yes")
        self.assertEqual(code, 0, output)
        self.assertFalse((self.root / "target.a3m").exists())

    def test_query_removal_requires_all(self):
        code, output = self.run_clean("--remove-query-msa", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("requires --all", output)
        self.assertTrue((self.root / "target.a3m").exists())

    def test_dry_run_does_not_mutate_or_query_slurm(self):
        before = sorted(self.root.rglob("*"))
        with patch.object(workflow, "active_jobs", side_effect=AssertionError("scheduler called")):
            code, output = self.run_clean("--dry-run", "--all", "--remove-query-msa")
        self.assertEqual(code, 0, output)
        self.assertIn("DELETE", output)
        self.assertEqual(before, sorted(self.root.rglob("*")))

    def test_shell_wrapper_works_from_another_directory(self):
        result = subprocess.run(["bash", str(REPO / "clean_screen.sh"), str(self.source), "--dry-run"],
                                cwd=self.base, capture_output=True, text=True,
                                env={**os.environ, "PYTHON_BIN": sys.executable})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.root), result.stdout)

    def test_delete_alias_uses_same_cleanup_options(self):
        result = subprocess.run([sys.executable, str(REPO / "latentppi.py"), "delete",
                                 str(self.source), "--all", "--remove-query-msa", "--dry-run"],
                                cwd=self.base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.root / "target.a3m"), result.stdout)
        self.assertTrue((self.root / "target.a3m").exists())

    def test_cancel_and_eof_leave_results(self):
        for answer in ("n", "", EOFError()):
            with patch("builtins.input", side_effect=answer if isinstance(answer, Exception) else None,
                       return_value=answer):
                code, output = self.run_clean()
            self.assertEqual(code, 0, output)
            self.assertIn("Aborted", output)
            self.assertTrue((self.root / "af3_predictions/pair/result.json").exists())

    def test_active_jobs_and_unavailable_scheduler_block_deletion(self):
        for response in (([{"job_id": "123"}], None), ([], "squeue unavailable")):
            with patch.object(workflow, "active_jobs", return_value=response):
                code, output = self.run_clean("--yes")
            self.assertEqual(code, 1, output)
            self.assertTrue((self.root / "af3_predictions/pair/result.json").exists())
            self.assertTrue((self.root / "latentppi_runs").exists())

    def test_real_ledger_requires_scheduler_and_corrupt_ledger_blocks(self):
        ledger = self.put("latentppi_runs/run-old/submissions.jsonl", json.dumps({"job_id": "123"}) + "\n")
        with patch.object(workflow.shutil, "which", return_value=None):
            code, output = self.run_clean("--yes")
        self.assertEqual(code, 1)
        self.assertIn("squeue unavailable", output)
        ledger.write_text("invalid json\n")
        code, output = self.run_clean("--yes")
        self.assertEqual(code, 1)
        self.assertIn("submission ledger", output)
        self.assertTrue(ledger.exists())

    def test_parent_cleanup_checks_advance_batch_before_deleting_shared_inputs(self):
        batch = self.root / "advance/batch-child"
        self.put("advance/batch-child/latentppi_runs/run-1/submissions.jsonl", '{"job_id":"456"}')
        def active(config):
            return ([{"job_id": "456"}] if Path(config.inputs.output_base) == batch else [], None)
        with patch.object(workflow, "active_jobs", side_effect=active):
            code, output = self.run_clean("--all", "--yes")
        self.assertEqual(code, 1, output)
        self.assertIn("456", output)
        self.assertTrue((self.root / "af3_inputs/pair.json").exists())

    def test_cleanup_respects_watcher_and_child_submission_locks(self):
        for directory, filename in ((self.root, ".latentppi_watch.lock"),
                                    (self.root / "advance/batch-child", ".latentppi_submit.lock")):
            with self.subTest(lock=directory / filename):
                with workflow.submission_lock(directory, filename=filename):
                    code, output = self.run_clean("--all", "--yes")
                self.assertEqual(code, 1, output)
                self.assertTrue((self.root / "af3_inputs/pair.json").exists())

    def test_job_recheck_after_confirmation(self):
        with patch.object(workflow, "active_jobs", side_effect=[([], None), ([{"job_id": "456"}], None)]), \
                patch("builtins.input", return_value="yes"):
            code, output = self.run_clean()
        self.assertEqual(code, 1, output)
        self.assertTrue((self.root / "af3_predictions").exists())

    def test_submission_lock_blocks_cleanup_and_keeps_inode(self):
        with workflow.submission_lock(self.root):
            inode = (self.root / ".latentppi_submit.lock").stat().st_ino
            code, output = self.run_clean("--yes")
            self.assertEqual(code, 1)
            self.assertIn("submission", output)
        self.assertEqual(self.run_clean("--yes")[0], 0)
        self.assertEqual(inode, (self.root / ".latentppi_submit.lock").stat().st_ino)

    def test_custom_output_paths_within_screen(self):
        self.raw["paths"] = dict(af3_output_dir="custom_predictions", ipsae_summary="scores/result.csv")
        self.raw["output"] = dict(metrics_dir="tables")
        self.write_config()
        for name in ("custom_predictions/result.json", "scores/result.csv", "scores/result.csv.lock",
                     "tables/metrics_all.csv", "tables/notes.txt"):
            self.put(name)
        code, output = self.run_clean("--yes")
        self.assertEqual(code, 0, output)
        self.assertFalse((self.root / "custom_predictions").exists())
        self.assertFalse((self.root / "scores/result.csv").exists())
        self.assertFalse((self.root / "tables/metrics_all.csv").exists())
        self.assertTrue((self.root / "tables/notes.txt").exists())

    def test_external_output_and_root_targets_rejected(self):
        for value in ("../external", "."):
            self.raw["paths"] = dict(af3_output_dir=value)
            self.write_config()
            code, output = self.run_clean("--yes")
            self.assertEqual(code, 1, output)
            self.assertIn("strictly inside", output)
            self.assertTrue((self.root / "af3_predictions").exists())

    def test_protected_resource_overlaps_rejected(self):
        cases = [("inputs", "proteome_msa_dir", str(self.root / "h5_shards")),
                 ("inputs", "fasta_file", str(self.root / "af3_predictions/source.fasta")),
                 ("paths", "summary_tsv", str(self.root / "h5_shards/source.tsv")),
                 ("scorer", "manifest_path", str(self.root / "latentppi_runs/scorer.json"))]
        original = json.loads(json.dumps(self.raw))
        for section, key, value in cases:
            self.raw = json.loads(json.dumps(original))
            self.raw.setdefault(section, {})[key] = value
            self.write_config()
            code, output = self.run_clean("--all", "--yes")
            self.assertEqual(code, 1, output)
            self.assertIn("protected", output)

    def test_configured_and_nested_symlinks_rejected(self):
        link = self.root / "alias"
        link.symlink_to(self.root / "af3_predictions", target_is_directory=True)
        self.raw["paths"] = dict(af3_output_dir="alias")
        self.write_config()
        code, output = self.run_clean("--yes")
        self.assertEqual(code, 1)
        self.assertIn("symlink", output)
        del self.raw["paths"]
        self.write_config()
        (self.root / "af3_predictions/shared").symlink_to(self.base / "proteome", target_is_directory=True)
        code, output = self.run_clean("--yes")
        self.assertEqual(code, 1)
        self.assertIn("symlink", output)
        self.assertTrue((self.base / "proteome").is_dir())

    def test_deletion_failure_retains_history_and_reports_partial_cleanup(self):
        with patch.object(clean.shutil, "rmtree", side_effect=PermissionError("denied")):
            code, output = self.run_clean("--yes")
        self.assertEqual(code, 1)
        self.assertIn("cleanup may be partial", output)
        self.assertTrue((self.root / "latentppi_runs").is_dir())

    def test_nonexistent_screen_dry_run_creates_nothing(self):
        self.raw["inputs"]["output_base"] = "absent"
        self.write_config()
        code, output = self.run_clean("--dry-run")
        self.assertEqual(code, 0, output)
        self.assertFalse((self.base / "absent").exists())

    def test_fallback_query_cache_removed_only_when_requested(self):
        digest = hashlib.sha256(str((self.base / "target.fasta").resolve()).encode()).hexdigest()[:12]
        cache = self.put(f"target_{digest}_query.a3m")
        other = self.put("unrelated.a3m")
        self.assertEqual(self.run_clean("--all", "--yes")[0], 0)
        self.assertTrue(cache.exists())
        self.assertEqual(self.run_clean("--all", "--remove-query-msa", "--yes")[0], 0)
        self.assertFalse(cache.exists())
        self.assertTrue(other.exists())

    def test_wrong_artifact_type_rejected(self):
        path = self.root / "metrics/metrics_all.csv"
        path.unlink()
        path.mkdir()
        (path / "valuable.txt").write_text("keep")
        code, output = self.run_clean("--yes")
        self.assertEqual(code, 1)
        self.assertIn("expected generated file", output)
        self.assertTrue((path / "valuable.txt").is_file())

    def test_unsafe_root_and_invalid_config_rejected(self):
        self.raw["inputs"]["output_base"] = str(REPO)
        self.write_config()
        code, output = self.run_clean("--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("unsafe screen root", output)
        self.source.write_text("inputs: null\n")
        code, output = self.run_clean("--yes")
        self.assertEqual(code, 1)
        self.assertIn("ERROR", output)


if __name__ == "__main__":
    unittest.main()
